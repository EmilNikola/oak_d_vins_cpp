/*
FPS(kamera, gyroskop a akcelerometr)xPERFORMANCE(filtry, NO of features, disparity)

CONFIG: projit funkce a nastaveni Stereodepth nastaveni, to je asi to hlavni - filtry, confidence threshold
CONFIG: jak se pristupuje ke zpracovani dat z IMU
CONFIG: report rate na senzorech porovnat s datasheetovymi specifikacemi, take rate jak jsou data odesilana nejak probrat
CONFIG: ktera z tech mereni latenci ukazuje to co chci?
CONFIG: jiz je pravdepodobne vse optimalni ale nechci odesilat jeste nejake jine zpravy nez ted?

COLOR CAMERA:   IMX378  4056x3040   85@2024x1520
MONO CAMERA:    OV9282  1280x800    THE_400_P: 255@640x400  THE_720_P: 143@1280x720 THE_800_P 129@1280x800  anti-banding mode*  3a algoritmy*

latency with LR and subpixel according to documentation: 800P: 30.5ms 400P: 10.1ms
*/

#include <iostream>
#include <thread>
#include <chrono>
#include <math.h>

#include <arpa/inet.h> // definitions for internet operations
#include <errno.h> // error indication
#include <stdio.h>
#include <sys/types.h> // data types for working with processes
#include <sys/socket.h> // communication endpoints
#include <unistd.h> // posix api
#include <string.h>
#include <sys/stat.h> // file metadata
#include <fcntl.h> // file descriptors manipulation
#include <sys/un.h> // unix sockets
#include <signal.h> // signal handling

// computer vision
#include <opencv2/calib3d.hpp>

// Includes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"
#include "deque"
#include "unordered_map"
#include "unordered_set"

// camera parameters as specified by THE_400_P
#define CAM_W 640
#define CAM_H 400
#define PAIR_DIST_SQ 9 // threshold macro
#define MAXIMUM_FEATURES 118
#define FPS 20

// 2D point location values
struct MyPoint2d {
    double x = 0;
    double y = 0;
    MyPoint2d() {}
    MyPoint2d(double px, double py) {
        x = px;
        y = py;
    }
};

double big_buf[12*1024/sizeof(double)]; // buffer size is just "big enough"
bool camera_run = true;

void sig_func(int sig) {
    camera_run = false;
}

void calc_rect_cam_intri_extri(dai::CalibrationHandler calibData, double* f, double* cx, double* cy) {
    std::cout << "stereo baseline:" << calibData.getBaselineDistance(dai::CameraBoardSocket::CAM_B, dai::CameraBoardSocket::CAM_C, false) << " cm\n CAMERA TO IMU EXTRINSICS:\n";
    // to make this available, IMU calibration data would need to be available at the time of calling this function, it seems unimportant at this moment
    /*auto imu_ext = calibData.getCameraToImuExtrinsics(dai::CameraBoardSocket::CAM_B, true);
    for (auto& row : imu_ext) {
        for (float val: row) {
            std::cout << "param: " << val << "\n";
        }
        std::cout << "\n";
    }*/

    auto l_intrinsics = calibData.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B, CAM_W, CAM_H);
    float data[9];
    int i = -1;
    for (auto row : l_intrinsics) {
        for (auto val : row) {
            data[++i] = val;
        }
    }
    cv::Mat l_m = cv::Mat(3, 3, CV_32FC1, data); // 3x3 matrix of instrinsics as 32bit float

    auto r_intrinsics = calibData.getCameraIntrinsics(dai::CameraBoardSocket::CAM_C, CAM_W, CAM_H);
    i = -1;
    for (auto row : r_intrinsics) {
        for (auto val : row) {
            data[++i] = val;
        }
    }
    cv::Mat r_m = cv::Mat(3, 3, CV_32FC1, data); // 3x3 matrix of instrinsics as 32bit float
    std::cout << "camera intrinsics left\n" << l_m << "\n right \n" << r_m << "\n"; // additional temporary(?) printout

    auto l_d = calibData.getDistortionCoefficients(dai::CameraBoardSocket::CAM_B);
    auto r_d = calibData.getDistortionCoefficients(dai::CameraBoardSocket::CAM_C);
    auto extrinsics = calibData.getCameraExtrinsics(dai::CameraBoardSocket::CAM_B, dai::CameraBoardSocket::CAM_C);

    cv::Mat r = (cv::Mat_<double>(3,3) << extrinsics[0][0], extrinsics[0][1], extrinsics[0][2], extrinsics[1][0], extrinsics[1][1], extrinsics[1][2], extrinsics[2][0], extrinsics[2][1], extrinsics[2][2]);
    cv::Mat t = (cv::Mat_<double>(3,1) << extrinsics[0][3], extrinsics[1][3], extrinsics[2][3]);
    std::cout << "stereo extrinsics\n" << r << "\n" << t << "\n"; // additional temporary(?) printout
    cv::Mat r1, r2, p1, p2, q;
    cv::stereoRectify(l_m, l_d, r_m, r_d, cv::Size(CAM_W, CAM_H), r, t, r1, r2, p1, p2, q, cv::CALIB_ZERO_DISPARITY, 0); // rectification transforms for stereo images alignment

    std::cout << "P1\n" << p1 << "\nP2\n" << p2 << "\n";

    *f = p1.at<double>(0, 0);
    *cx = p1.at<double>(0, 2);
    *cy = p1.at<double>(1, 2);
}

int main(int argc, char **argv) {
    bool imu_ok = false;
    int ccc=0; // number of frames

    // terminate process by calling SIGINT(Ctrl-C)
    struct sigaction act;
    memset(&act, 0, sizeof(act));
    act.sa_handler = sig_func; // pointer to function
    sigaction(SIGINT, &act, NULL);

    // creating unix sockets, ipc_local_addr to receive and other two to send data
    struct sockaddr_un ipc_local_addr, imu_addr, features_addr;
    memset(&ipc_local_addr, 0, sizeof(struct sockaddr_un));
    ipc_local_addr.sun_family = AF_UNIX;
    strcpy(ipc_local_addr.sun_path, "/tmp/chobits_2222");
    int ipc_sock = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (ipc_sock < 0) {
        perror("ipc_sock creation failed");
        exit(EXIT_FAILURE);
    }
    unlink("/tmp/chobits_2222");
    if (bind(ipc_sock, (struct sockaddr*)&ipc_local_addr, sizeof(ipc_local_addr)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    memset(&imu_addr, 0, sizeof(struct sockaddr_un));
    imu_addr.sun_family = AF_UNIX;
    strcpy(imu_addr.sun_path, "/tmp/chobits_imu");
    memset(&features_addr, 0, sizeof(struct sockaddr_un));
    features_addr.sun_family = AF_UNIX;
    strcpy(features_addr.sun_path, "/tmp/chobits_features");

    // Create pipeline
    dai::Pipeline pipeline;

    // Define sources and outputs
    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    //auto colorCam = pipeline.create<dai::node::ColorCamera>();
    auto featureTrackerLeft = pipeline.create<dai::node::FeatureTracker>();
    auto featureTrackerRight = pipeline.create<dai::node::FeatureTracker>();
    auto imu = pipeline.create<dai::node::IMU>();

    auto xoutTrackedFeaturesLeft = pipeline.create<dai::node::XLinkOut>();
    auto xoutTrackedFeaturesRight = pipeline.create<dai::node::XLinkOut>();
    auto xinTrackedFeaturesConfig = pipeline.create<dai::node::XLinkIn>();
    auto depth = pipeline.create<dai::node::StereoDepth>();
    auto xout_disp = pipeline.create<dai::node::XLinkOut>();
    auto xout_imu = pipeline.create<dai::node::XLinkOut>();
    // auto xout_focal = pipeline.create<dai::node::XLinkOut>(); //here

    // specify some stream names over which nodes receive their data
    xoutTrackedFeaturesLeft->setStreamName("trackedFeaturesLeft");
    xoutTrackedFeaturesRight->setStreamName("trackedFeaturesRight");
    xinTrackedFeaturesConfig->setStreamName("trackedFeaturesConfig");
    xout_disp->setStreamName("disparity");
    xout_imu->setStreamName("imu");
    //xout_focal->setStreamName("focal"); //here

    // Properties
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoLeft->setCamera("left");
    monoLeft->setFps(FPS);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setCamera("right");
    monoRight->setFps(FPS);

    featureTrackerLeft->initialConfig.setNumTargetFeatures(16*5);
    featureTrackerRight->initialConfig.setNumTargetFeatures(16*5);
    // Initialize motion estimator to hardware-accelerated mode (can be changed at runtime via trackedFeaturesConfig)
    {
        auto ftCfg = featureTrackerLeft->initialConfig.get();
        ftCfg.motionEstimator.type = dai::FeatureTrackerConfig::MotionEstimator::Type::HW_MOTION_ESTIMATION;
        featureTrackerLeft->initialConfig.set(ftCfg);
        featureTrackerRight->initialConfig.set(ftCfg);
    }
    /*dai::RawFeatureTrackerConfig config = featureTrackerLeft->initialConfig.get();
    config.cornerDetector.numMaxFeatures = 100;
    featureTrackerLeft->initialConfig.set(config);
    config = featureTrackerRight->initialConfig.get();
    config.cornerDetector.numMaxFeatures = 100;
    featureTrackerRight->initialConfig.set(config);*/
    
    // according to API refrence for both Shaves and Memory slices, maximum number is allocated
    featureTrackerLeft->setHardwareResources(2, 2);
    featureTrackerRight->setHardwareResources(2, 2);

    depth->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_ACCURACY);
    depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_5x5);
    depth->setLeftRightCheck(true);
    depth->setExtendedDisparity(false);
    depth->setSubpixel(true);
    depth->setSubpixelFractionalBits(3); 
    depth->setDepthAlign(dai::RawStereoDepthConfig::AlgorithmControl::DepthAlign::RECTIFIED_LEFT);
    depth->setAlphaScaling(0);

    // enable ACCELEROMETER_RAW at 500 hz rate
    imu->enableIMUSensor(dai::IMUSensor::ACCELEROMETER, 125);
    // enable GYROSCOPE_RAW at 400 hz rate
    imu->enableIMUSensor(dai::IMUSensor::GYROSCOPE_CALIBRATED, 100);
    // it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
    // above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
    imu->setBatchReportThreshold(1);
    // maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
    // if lower or equal to batchReportThreshold then the sending is always blocking on device
    // useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
    imu->setMaxBatchReports(10);

    // Linking
    monoLeft->out.link(depth->left);
    depth->rectifiedLeft.link(featureTrackerLeft->inputImage);
    featureTrackerLeft->outputFeatures.link(xoutTrackedFeaturesLeft->input);

    monoRight->out.link(depth->right);
    depth->rectifiedRight.link(featureTrackerRight->inputImage);
    featureTrackerRight->outputFeatures.link(xoutTrackedFeaturesRight->input);

    depth->disparity.link(xout_disp->input);
    imu->out.link(xout_imu->input);
    // Allow runtime updates to FeatureTracker configuration for both left and right
    xinTrackedFeaturesConfig->out.link(featureTrackerLeft->inputConfig);
    xinTrackedFeaturesConfig->out.link(featureTrackerRight->inputConfig);
    // Link color camera ISP output to focal XLinkOut so we get lens metadata
    // colorCam->setPreviewSize(CAM_W, CAM_H);
    // colorCam->setFps(20);
    // Ensure 3A runs at camera framerate so AF metadata is available
    // colorCam->setIsp3aFps(20);
    // colorCam->isp.link(xout_focal->input); // send ISP frames from color camera to host

    // list cameras so i know which one to configure
    std::cout << "Searching for all available devices...\n\n";
    auto infos = dai::Device::getAllAvailableDevices();
    if(infos.empty()) {
        std::cout << "Couldn't find any available devices.\n";
        return -1;
    }
    for(auto& info : infos) {
        std::cout << "Found device: " << info.name << " mxid: " << info.mxid << " state: " << info.state << "\n";
    }

    // Connect to device and start pipeline
    dai::Device device(pipeline);

    // device parameters readout
    std::cout << "Usb speed: " << device.getUsbSpeed() << "\n";
    std::cout << "Device name: " << device.getDeviceName() << " Product name: " << device.getProductName() << "\n";

    dai::CalibrationHandler calibData = device.readCalibration2();
    double f, cx, cy;
    float baseline = calibData.getBaselineDistance(dai::CameraBoardSocket::CAM_B, dai::CameraBoardSocket::CAM_C, false) * 0.01f;
    calc_rect_cam_intri_extri(calibData, &f, &cx, &cy);
    float hfov = 2 * atanf(CAM_W / (2 * f));
    float vfov = 2 * atanf(CAM_H / (2 * f));
    std::cout << "stereo baseline:" << baseline << " m, f:" << f << " px, cx:" << cx << ", cy:" << cy << " hfov:" << hfov * 180 / M_PI << " degrees, vfov:" << vfov * 180 / M_PI << " degrees\n";

    // variables for affine transformation
    double l_inv_k11 = 1.0 / f; // inverse focal length - diopters
    double l_inv_k13 = -cx / f; // horizontal center/focal length
    double l_inv_k22 = 1.0 / f;
    double l_inv_k23 = -cy / f; // vertical center/focal length
    double r_inv_k11 = 1.0 / f;
    double r_inv_k13 = -cx / f;
    double r_inv_k22 = 1.0 / f;
    double r_inv_k23 = -cy / f;

    auto s_pairs = device.getAvailableStereoPairs();
    for (auto& s_pair : s_pairs) {
        std::cout << "(TEMPORARY PRINTOUT: possilbe stereo pair baseline:" << s_pair.baseline << " cm\n";
    }

    // verbose logging
    //device.setLogOutputLevel(dai::LogLevel::DEBUG);
    //device.setLogLevel(dai::LogLevel::DEBUG);

    // Output queues used to receive the results
    // 3rd argument when false specifies that old messages are overwritten when the queue is full
    auto outputFeaturesLeftQueue = device.getOutputQueue("trackedFeaturesLeft", 1, false);
    auto outputFeaturesRightQueue = device.getOutputQueue("trackedFeaturesRight", 1, false);
    auto disp_queue = device.getOutputQueue("disparity", 1, false);
    auto imuQueue = device.getOutputQueue("imu", 5, false);
    // Input queue to send runtime FeatureTracker config updates (optional)
    auto inputFeatureTrackerConfigQueue = device.getInputQueue("trackedFeaturesConfig");
    // auto focalQueue = device.getOutputQueue("focal", 1, false); //here

    // sequence numbers initialisation
    int l_seq = -1, r_seq = -2, disp_seq = -3;
    // int64_t prev_lens_pos = -1000; //here
    // float prev_lens_pos_raw = -1.0f;//here

    // tools for variable processing
    std::vector<std::uint8_t> disp_frame; // disparity frame data
    uint16_t* pDisp_frame16 = nullptr; // pointer to transdormed disparity frame data
    std::vector<dai::TrackedFeature> l_features, r_features; // vectors containing features
    std::unordered_map<int, dai::Point2f> l_prv_features, r_prv_features; // vectors containing features from previous frame
    std::unordered_map<int, dai::Point2f> r_cur_features; // right image current features indexed map
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::steady_clock::duration> features_tp, prv_features_tp; // timestamps of frames
    std::unordered_map<int, int> lr_id_mapping; // features detected in left image paired to features in right

    // Clear queue events
    //jakaskerl suggest remove this line
    //https://discuss.luxonis.com/d/3484-getqueueevent-takes-much-additional-time/7
    //device.getQueueEvents();

    float l_sum = 0.0, r_sum = 0.0, disp_sum = 0.0;
    int l_count = 0, r_count = 0, disp_count = 0;

    while(camera_run) {
        auto q_name = device.getQueueEvent();

        if (q_name == "trackedFeaturesLeft") { // waits until specified queue gets a message
            auto data = outputFeaturesLeftQueue->get<dai::TrackedFeatures>();
            l_features = data->trackedFeatures;
            l_seq = data->getSequenceNum(); // retrieve sequence number
            features_tp = data->getTimestampDevice(); // timestamp from camera
            //std::cout << "LEFT ft " << l_seq << " latency:" << std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - features_tp).count() << " ms\n";
            l_sum += std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - features_tp).count();
            l_count += 1;
        } else if (q_name == "trackedFeaturesRight") {
            auto data = outputFeaturesRightQueue->get<dai::TrackedFeatures>();
            r_features = data->trackedFeatures;
            r_seq = data->getSequenceNum();
            //std::cout << "RIGHT ft " << r_seq << " latency:" << std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - data->getTimestamp()).count() << " ms\n";
            r_sum += std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - data->getTimestamp()).count();
            r_count += 1;
            r_cur_features.clear();
            for (const auto &feature : r_features) {
                r_cur_features[feature.id] = feature.position; // map features to indexes
            }
        } else if (q_name == "disparity") {
            auto disp_data = disp_queue->get<dai::ImgFrame>();
            disp_seq = disp_data->getSequenceNum();
            disp_frame = disp_data->getData(); // return only disparity data from frame
            pDisp_frame16 = (uint16_t*)disp_frame.data();
            // std::cout << "stereo " << disp_seq << " latency:" << std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - disp_data->getTimestamp()).count() << " ms\n";
            disp_sum += std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - disp_data->getTimestamp()).count();
            disp_count += 1;
        /*} else if (q_name == "focal") {
            auto data = focalQueue->get<dai::ImgFrame>();
            // sequence number available if needed
            //int cam_seq = cam_frame->getSequenceNum();
            // very likely wrong camera, need to run on camera 1st
            int lens_pos = cam_frame->getLensPosition(dai::CameraBoardSocket::CAM_A); // 0..255 or -1 if not available
            float lens_pos_raw = cam_frame->getLensPositionRaw(dai::CameraBoardSocket::CAM_A); // 0.0..1.0 or -1 if not available
            // auto ts = cam_frame->getTimestampDevice();
            if (lens_pos != prev_lens_pos) {
                std::cout << " lens_pos=" << lens_pos << " lens_pos_raw=" << lens_pos_raw << "\n";
                prev_lens_pos = lens_pos;
                //prev_lens_pos_raw = lens_pos_raw;
            }*/
        } else if (q_name == "imu") {
            auto imuData = imuQueue->get<dai::IMUData>();
            auto imuPackets = imuData->packets;
            for(const auto& imuPacket : imuPackets) {
                auto& acc = imuPacket.acceleroMeter;
                auto& gyro = imuPacket.gyroscope;
                //std::cout << "imu latency, acc:" << std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - acc.getTimestamp()).count() << " ms, gyro:" << std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - gyro.getTimestamp()).count() << " ms\n";
                big_buf[0] = std::chrono::duration<double>(gyro.getTimestampDevice().time_since_epoch()).count();
                // translate to ros frame, easier to understand in rviz
                big_buf[1] = -acc.z;
                big_buf[2] = -acc.y;
                big_buf[3] = -acc.x;
                big_buf[4] = -gyro.z;
                big_buf[5] = -gyro.y;
                big_buf[6] = -gyro.x;
                sendto(ipc_sock, big_buf, 7*sizeof(double), 0, (struct sockaddr*)&imu_addr, sizeof(struct sockaddr_un));
            }
            if (!imu_ok) {
                imu_ok = true;
                std::cout << "imu ok\n";
            }
        }

        if (l_seq == r_seq && r_seq == disp_seq) { // executes if left, right and disparity frames align
            //auto t1 = std::chrono::steady_clock::now();
            l_seq = -1;
            r_seq = -2;
            disp_seq = -3;
            std::unordered_map<int , dai::Point2f> features;
            int c = 0;
            big_buf[1] = std::chrono::duration<double>(features_tp.time_since_epoch()).count(); // duration between epoch and timestamp of frame expressed in seconds
            double* buf_ptr = big_buf + 2; // first two slots are occupied with timestamps
            for (const auto &l_feature : l_features) {
                float x = l_feature.position.x;
                float y = l_feature.position.y;
                double cur_un_x = l_inv_k11 * x + l_inv_k13; // normalised current x position
                double cur_un_y = l_inv_k22 * y + l_inv_k23; // normalised current y position
                features[l_feature.id] = dai::Point2f(static_cast<float>(cur_un_x), static_cast<float>(cur_un_y)); // map of normalised values
                auto lr_id = lr_id_mapping.find(l_feature.id); // check if feature is one that has a match
                if (lr_id != lr_id_mapping.end()) { // checks if previous line found instance
                    auto r_feature = r_cur_features.find(lr_id->second); // tries to find that feature in current right side
                    if (r_feature != r_cur_features.end()) { // checks if previous line found instance
                        double dt = std::chrono::duration<double>(features_tp - prv_features_tp).count(); // timestamp difference between current and latest dispartity frame
                        double vx = 0, vy = 0;
                        auto prv_pos = l_prv_features.find(l_feature.id); // checks if previous left side feature is located in the new one
                        if (prv_pos != l_prv_features.end()) { // if its found, normalised speed is calculated
                            vx = (cur_un_x - prv_pos->second.x) / dt;
                            vy = (cur_un_y - prv_pos->second.y) / dt;
                        }
                        buf_ptr[0] = l_feature.id; // store id
                        buf_ptr[1] = cur_un_x; // store normalised position x
                        buf_ptr[2] = cur_un_y; // and y
                        buf_ptr[3] = x; // store position x
                        buf_ptr[4] = y; // and y
                        buf_ptr[5] = vx; // store x 
                        buf_ptr[6] = vy; // and y speed
                        // storing right feature positions instead of left
                        x = r_feature->second.x;
                        y = r_feature->second.y;
                        vx = 0;
                        vy = 0;
                        cur_un_x = r_inv_k11 * x + r_inv_k13;
                        cur_un_y = r_inv_k22 * y + r_inv_k23;
                        prv_pos = r_prv_features.find(r_feature->first);
                        if (prv_pos != r_prv_features.end()) {
                            vx = (cur_un_x - prv_pos->second.x) / dt;
                            vy = (cur_un_y - prv_pos->second.y) / dt;
                        }
                        buf_ptr[7] = cur_un_x;
                        buf_ptr[8] = cur_un_y;
                        buf_ptr[9] = x;
                        buf_ptr[10] = y;
                        buf_ptr[11] = vx;
                        buf_ptr[12] = vy;

                        if (c < MAXIMUM_FEATURES) { // maximum number of features
                            ++c;
                            buf_ptr += 13; // move to next position in buffer accordingly
                        }

                        continue;
                    }
                }
                // rounding down 
                int col = roundf(x);
                int row = roundf(y);
                // setting bounds for possible values
                if (col > CAM_W - 1) col = CAM_W - 1;
                if (row > CAM_H - 1) row = CAM_H - 1;
                float disp = pDisp_frame16[row * CAM_W + col] / 8.0f; // disparity value at pixel position
                if (disp > 0) { // if there exists a disparity
                    for (const auto &r_feature : r_features) {
                        float dy = y - r_feature.position.y; // difference between l and accredited to noise
                        float dx = x - disp - r_feature.position.x; // difference = noise and also disparity (epipolar shift)
                        if (dy * dy + dx * dx <= PAIR_DIST_SQ) { //pair found, aim for 95 percentile?
                            lr_id_mapping[l_feature.id] = r_feature.id; // persists over multiple frames if feature is repeatedly found
                            // same as left side
                            double dt = std::chrono::duration<double>(features_tp - prv_features_tp).count();
                            double vx = 0, vy = 0;
                            auto prv_pos = l_prv_features.find(l_feature.id);
                            if (prv_pos != l_prv_features.end()) {
                                vx = (cur_un_x - prv_pos->second.x) / dt;
                                vy = (cur_un_y - prv_pos->second.y) / dt;
                            }
                            buf_ptr[0] = l_feature.id;
                            buf_ptr[1] = cur_un_x;
                            buf_ptr[2] = cur_un_y;
                            buf_ptr[3] = x;
                            buf_ptr[4] = y;
                            buf_ptr[5] = vx;
                            buf_ptr[6] = vy;

                            x = r_feature.position.x;
                            y = r_feature.position.y;
                            vx = 0;
                            vy = 0;
                            cur_un_x = r_inv_k11 * x + r_inv_k13;
                            cur_un_y = r_inv_k22 * y + r_inv_k23;
                            prv_pos = r_prv_features.find(r_feature.id);
                            if (prv_pos != r_prv_features.end()) {
                                vx = (cur_un_x - prv_pos->second.x) / dt;
                                vy = (cur_un_y - prv_pos->second.y) / dt;
                            }
                            buf_ptr[7] = cur_un_x;
                            buf_ptr[8] = cur_un_y;
                            buf_ptr[9] = x;
                            buf_ptr[10] = y;
                            buf_ptr[11] = vx;
                            buf_ptr[12] = vy;

                            if (c < 118) {
                                ++c;
                                buf_ptr += 13;
                            }

                            break;
                        }
                    }
                }
            }
            // logging every ccc=60 frames
            ccc++;
            if (ccc > 60) {
                ccc = 0;
                std::cout << c << " features\n";
                std::cout << "average latency LEFT: " << l_sum/l_count << " ms\n";
                std::cout << "average latency RIGHT: " << r_sum/r_count << " ms\n";
                std::cout << "average latency DISPARITY: " << disp_sum/disp_count << " ms\n";
                l_sum = 0.0;
                r_sum = 0.0;
                disp_sum = 0.0;
                l_count = 0;
                r_count = 0;
                disp_count = 0;
            }
            if (c < 10) std::cout << "WARNING: too few features: " << c << "\n";
            // sending features
            if (imu_ok && c > 0) {
                big_buf[0] = c;
                sendto(ipc_sock, big_buf, 13*sizeof(double)*c+2*sizeof(double), 0, (struct sockaddr*)&features_addr, sizeof(struct sockaddr_un));
            }

            // current frame moved to previous to make place for new frame
            l_prv_features = features;
            prv_features_tp = features_tp;
            r_prv_features.clear();
            for (const auto &r_feature : r_features) {
                r_prv_features[r_feature.id] = dai::Point2f(
                    static_cast<float>(r_inv_k11 * r_feature.position.x + r_inv_k13),
                    static_cast<float>(r_inv_k22 * r_feature.position.y + r_inv_k23)
                );
            }
            //auto t2 = std::chrono::steady_clock::now();
            //std::cout << pp_msg.points.size() << " points, " << std::chrono::duration<float, std::milli>(t2-t1).count() << " ms\n";
        }
    }

    close(ipc_sock);
    std::cout << "bye\n";

    return 0;
}
