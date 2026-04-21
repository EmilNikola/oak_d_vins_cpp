/*
odstranit potencialni deleni nulou (rychlosti)
je pDisp_frame16 safe? k zamysleni

NUTNOST KAMERY:
CONFIG: jestli se CV rectification projevi jako zbytecna, bude odstranena pro rychlejsi loop - nahrazeni setRectification(True)

COLOR CAMERA:   IMX378  4056x3040   85@2024x1520
MONO CAMERA:    OV9282  1280x800    THE_400_P: 255@640x400  THE_720_P: 143@1280x720 THE_800_P 129@1280x800  anti-banding mode*  3a algoritmy*

latency with LR and subpixel according to documentation: 800P: 30.5ms 400P: 10.1ms
*/

#include <iostream>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <math.h>
#include <stdio.h>
#include <sys/types.h> // data types for working with processes
#include <sys/socket.h> // communication endpoints
#include <unistd.h> // posix api
#include <string.h>
#include <string>
#include <sys/un.h> // unix sockets
#include <signal.h> // signal handling

// computer vision
#include <opencv2/calib3d.hpp>
// GUI / drawing
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>

// Includes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"
#include "deque"
#include "unordered_map"

// camera parameters as specified by THE_400_P
#define CAM_W 640
#define CAM_H 400
#define PAIR_DIST_SQ 9 // threshold macro
#define MIN_FEATURES 10
#define TARGET_FEATURES 80 // 320 is the default from source
#define MAXIMUM_FEATURES 118
#define FPS 20
#define FRAC_BITS_N 3
#define NUMBEROF_DATA 13

// 2D point location values
// struct MyPoint2d {
//     double x = 0;
//     double y = 0;
//     MyPoint2d() {}
//     MyPoint2d(double px, double py) {
//         x = px;
//         y = py;
//     }
// };

volatile sig_atomic_t camera_run = 1;
void sig_func(int sig) {
    camera_run = 0;
}

void calc_rect_cam_intri_extri(dai::CalibrationHandler calibData, double* f, double* cx, double* cy) {
    
    float data[9]; // left and right intrinsics

    // bool uses translation information from board design data
    // stereo baseline:7.50001stereo baseline:7.5 cm
    //std::cout << "stereo baseline:" << calibData.getBaselineDistance(dai::CameraBoardSocket::CAM_B, dai::CameraBoardSocket::CAM_C, false);
    //std::cout << "stereo baseline:" << calibData.getBaselineDistance(dai::CameraBoardSocket::CAM_B, dai::CameraBoardSocket::CAM_C, true) << " cm\n CAMERA TO IMU EXTRINSICS:\n";
    
    // to make this available, IMU calibration data would need to be available at the time of calling this function, it seems unimportant at this moment
    /*auto imu_ext = calibData.getCameraToImuExtrinsics(dai::CameraBoardSocket::CAM_B, true);
    for (auto& row : imu_ext) {
        for (float val: row) {
            std::cout << "param: " << val << "\n";
        }
        std::cout << "\n";
    }*/

    auto l_intrinsics = calibData.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B, CAM_W, CAM_H);
    int i = 0;
    for (auto row : l_intrinsics) {
        for (auto val : row) {
            data[i++] = val;
        }
    }
    cv::Mat intri_l = cv::Mat(3, 3, CV_32FC1, data); // 3x3 matrix of instrinsics as 32bit float

    auto r_intrinsics = calibData.getCameraIntrinsics(dai::CameraBoardSocket::CAM_C, CAM_W, CAM_H);
    i = 0;
    for (auto row : r_intrinsics) {
        for (auto val : row) {
            data[i++] = val;
        }
    }
    cv::Mat intri_r = cv::Mat(3, 3, CV_32FC1, data); // 3x3 matrix of instrinsics as 32bit float

    std::cout << "camera intrinsics left\n" << intri_l << "\n right \n" << intri_r << "\n"; // additional temporary(?) printout

    auto dist_l = calibData.getDistortionCoefficients(dai::CameraBoardSocket::CAM_B);
    auto dist_r = calibData.getDistortionCoefficients(dai::CameraBoardSocket::CAM_C);
    
    auto extrinsics = calibData.getCameraExtrinsics(dai::CameraBoardSocket::CAM_B, dai::CameraBoardSocket::CAM_C);
    cv::Mat r = (cv::Mat_<double>(3,3) << extrinsics[0][0], extrinsics[0][1], extrinsics[0][2], extrinsics[1][0], extrinsics[1][1], extrinsics[1][2], extrinsics[2][0], extrinsics[2][1], extrinsics[2][2]);
    cv::Mat t = (cv::Mat_<double>(3,1) << extrinsics[0][3], extrinsics[1][3], extrinsics[2][3]);
    std::cout << "stereo extrinsics\n" << r << "\n" << t << "\n"; // additional temporary(?) printout
    
    cv::Mat r1, r2, p1, p2, q;
    cv::stereoRectify(intri_l, dist_l, intri_r, dist_r, cv::Size(CAM_W, CAM_H), r, t, r1, r2, p1, p2, q, cv::CALIB_ZERO_DISPARITY, 0); // rectification transforms for stereo images alignment
    // p1 and p2 are projection matrices in rectified coordinate system for cameras
    // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
    std::cout << "P1\n" << p1 << "\nP2\n" << p2 << "\n";
    *f = p1.at<double>(0, 0);
    *cx = p1.at<double>(0, 2);
    *cy = p1.at<double>(1, 2);
}

void printConfig(const char *time, dai::RawFeatureTrackerConfig cfg) {
    std::cout << "Config " << time << " modification:\n";
    std::cout << "numTargetFeatures " << cfg.cornerDetector.numTargetFeatures << "\n";
    std::cout << "numMaxFeatures " << cfg.cornerDetector.numMaxFeatures << "\n";
    std::cout << "cornerDetectorType " << static_cast<int>(cfg.cornerDetector.type) << "\n";
    std::cout << "maintainTresholdsDistance " << cfg.featureMaintainer.minimumDistanceBetweenFeatures << "\n";
    std::cout << "maintainTresholdsTostFeature " << cfg.featureMaintainer.lostFeatureErrorThreshold << "\n";
    std::cout << "maintainTresholdsTrackedFeature " << cfg.featureMaintainer.trackedFeatureThreshold << "\n";
}

int main(int argc, char **argv) {
    bool imu_ok = false;
    int num_frames=0; // number of frames
    int allow = 0;
    if(argc > 10) {
        allow = static_cast<int>(strtol(argv[10], NULL, 10));
    }
    // depth preset selection via argv[4] (optional):
    // 0 = FAST_ACCURACY
    // 1 = FAST_DENSITY
    // 2 = DEFAULT
    // 3 = FACE
    // 4 = HIGH_DETAIL
    // 5 = ROBOTICS
    int depth_preset_idx = 0; // default
    if(argc > 4) {
        depth_preset_idx = static_cast<int>(strtol(argv[4], NULL, 10));
    }
    // optional overrides (argv indices after existing args):
    // argv[11] = median filter: 0=MEDIAN_OFF, 1=KERNEL_5x5, 2=KERNEL_7x7
    // argv[12] = extended disparity: 0=false, 1=true
    // argv[13] = subpixel enabled: 0=false, 1=true
    // argv[14] = subpixel fractional bits (int)
    int median_idx = -1;
    int ext_disp_idx = -1;
    int subpixel_flag = -1;
    int subpixel_frac = -1;
    int motion_idx = -1; // 0 = Lucas-Kanade (default), 1 = HW motion estimation
    if(argc > 11) median_idx = static_cast<int>(strtol(argv[11], NULL, 10));
    if(argc > 12) ext_disp_idx = static_cast<int>(strtol(argv[12], NULL, 10));
    if(argc > 13) subpixel_flag = static_cast<int>(strtol(argv[13], NULL, 10));
    if(argc > 14) subpixel_frac = static_cast<int>(strtol(argv[14], NULL, 10));
    if(argc > 15) motion_idx = static_cast<int>(strtol(argv[15], NULL, 10));

    // terminate process by calling SIGINT(Ctrl-C)
    struct sigaction act;
    memset(&act, 0, sizeof(act));
    act.sa_handler = sig_func; // pointer to function
    sigaction(SIGINT, &act, NULL);

    // creating unix socket ,ipc_local_addr to send and receive points features_addr, imu_addr
    struct sockaddr_un ipc_local_addr, imu_addr, features_addr;
    unlink("/tmp/chobits_2222");
    memset(&ipc_local_addr, 0, sizeof(struct sockaddr_un));
    ipc_local_addr.sun_family = AF_UNIX;
    strcpy(ipc_local_addr.sun_path, "/tmp/chobits_2222");
    int ipc_sock = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (ipc_sock == -1) {
        perror("ipc_sock creation failed");
        exit(EXIT_FAILURE);
    }
    if (bind(ipc_sock, (struct sockaddr*)&ipc_local_addr, sizeof(ipc_local_addr)) == -1) {
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
    auto depth = pipeline.create<dai::node::StereoDepth>();

    auto xoutTrackedFeaturesLeft = pipeline.create<dai::node::XLinkOut>();
    auto xoutTrackedFeaturesRight = pipeline.create<dai::node::XLinkOut>();
    // passthrough XLinkOuts: declare pointers here so they are in scope for later linking
    std::shared_ptr<dai::node::XLinkOut> xoutPassthroughFrameLeft = nullptr;
    std::shared_ptr<dai::node::XLinkOut> xoutPassthroughFrameRight = nullptr;
    if (allow == 1) {
        xoutPassthroughFrameLeft = pipeline.create<dai::node::XLinkOut>();
        xoutPassthroughFrameRight = pipeline.create<dai::node::XLinkOut>();
    }
    auto xinTrackedFeaturesConfig = pipeline.create<dai::node::XLinkIn>();
    auto xout_disp = pipeline.create<dai::node::XLinkOut>();
    auto xout_imu = pipeline.create<dai::node::XLinkOut>();

    // specify some stream names over which nodes receive their data
    xoutTrackedFeaturesLeft->setStreamName("trackedFeaturesLeft");
    xoutTrackedFeaturesRight->setStreamName("trackedFeaturesRight");
    if (allow == 1 && xoutPassthroughFrameLeft && xoutPassthroughFrameRight) {
        xoutPassthroughFrameLeft->setStreamName("passthroughFrameLeft");
        xoutPassthroughFrameRight->setStreamName("passthroughFrameRight");
    }
    xinTrackedFeaturesConfig->setStreamName("trackedFeaturesConfig");
    xout_disp->setStreamName("disparity");
    xout_imu->setStreamName("imu");

    // Properties
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoLeft->setCamera("left");
    monoLeft->setFps(strtol(argv[3],NULL,10));
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setCamera("right");
    monoRight->setFps(strtol(argv[3],NULL,10));


    // Initializes motion estimator (default: Lucas-Kanade)
    auto featureTrackerConfig = featureTrackerLeft->initialConfig.get();
    printConfig("before", featureTrackerConfig);

    featureTrackerConfig.cornerDetector.numTargetFeatures = strtol(argv[1],NULL,10);
    featureTrackerConfig.cornerDetector.numMaxFeatures = strtol(argv[2],NULL,10);


    //HARRIS OR SHI_THOMASI, I prefer shi_tomasi
    featureTrackerConfig.cornerDetector.type = dai::FeatureTrackerConfig::CornerDetector::Type::SHI_THOMASI;
    // Motion estimator selection: 0 = Lucas-Kanade (default), 1 = HW motion estimation
    if(motion_idx == 1) {
        featureTrackerConfig.motionEstimator.type = dai::FeatureTrackerConfig::MotionEstimator::Type::HW_MOTION_ESTIMATION;
    } else {
        featureTrackerConfig.motionEstimator.type = dai::FeatureTrackerConfig::MotionEstimator::Type::LUCAS_KANADE_OPTICAL_FLOW;
        // Lukas-Kanade empirical config (only relevant for Lucas-Kanade)
        featureTrackerConfig.motionEstimator.opticalFlow.searchWindowWidth = strtol(argv[7],NULL,10);
        featureTrackerConfig.motionEstimator.opticalFlow.searchWindowHeight = strtol(argv[7],NULL,10);
        featureTrackerConfig.motionEstimator.opticalFlow.epsilon = std::stof(argv[8], NULL);
        featureTrackerConfig.motionEstimator.opticalFlow.maxIterations = strtol(argv[9],NULL,10);
    }
    //featureMaintainer likely to remain unchanged
    //featureTrackerConfig.FeatureMaintainer.minimumDistanceBetweenFeatures = 50;
    //featureTrackerConfig.FeatureMaintainer.lostFeatureErrorThreshold = 50000;
    //featureTrackerConfig.FeatureMaintainer.trackedFeatureThreshold = 200000;
    

    featureTrackerLeft->initialConfig.set(featureTrackerConfig);
    featureTrackerRight->initialConfig.set(featureTrackerConfig);
    printConfig("after", featureTrackerConfig);

    // according to API refrence for both Shaves and Memory slices, maximum number is allocated
    auto numShaves = 2;
    auto numSlices = 2;
    featureTrackerLeft->setHardwareResources(numShaves, numSlices);
    featureTrackerRight->setHardwareResources(numShaves, numSlices);

    // choose preset parsed from argv[4]
    dai::node::StereoDepth::PresetMode depthPreset;
    switch(depth_preset_idx) {
        case 1:
            depthPreset = dai::node::StereoDepth::PresetMode::HIGH_ACCURACY;
            break;
        case 2:
            depthPreset = dai::node::StereoDepth::PresetMode::HIGH_DENSITY;
            break;
        default:
            depthPreset = dai::node::StereoDepth::PresetMode::HIGH_ACCURACY;
            break;
    }
    depth->setDefaultProfilePreset(depthPreset);

    // Apply optional overrides after preset so they overwrite preset values
    if(median_idx >= 0) {
        switch(median_idx) {
            case 0:
                depth->initialConfig.setMedianFilter(dai::MedianFilter::MEDIAN_OFF);
                break;
            case 1:
                depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_5x5);
                break;
            case 2:
                depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_7x7);
                break;
            default:
                depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_5x5);
                break;
        }
    }
    if(ext_disp_idx >= 0) {
        depth->setExtendedDisparity(ext_disp_idx != 0);
    }
    if(subpixel_flag >= 0) {
        depth->setSubpixel(subpixel_flag != 0);
    }
    if(subpixel_frac >= 0) {
        depth->setSubpixelFractionalBits(subpixel_frac);
    }

    depth->setDepthAlign(dai::RawStereoDepthConfig::AlgorithmControl::DepthAlign::RECTIFIED_LEFT); // not within preset
    depth->setAlphaScaling(0); // not within preset
    /*
    possibly beneficial but potentional issues
    - disparity indexing still works?
    - normalized math works?
    */

    // Accelerometer options: 15Hz, 31Hz, 62Hz, 125Hz, 250Hz 500Hz
    imu->enableIMUSensor(dai::IMUSensor::ACCELEROMETER, strtol(argv[5],NULL,10));
        // Gyroscope options: 25Hz, 33Hz, 50Hz, 100Hz, 200Hz, 400Hz; 100 might be the max for this option
    imu->enableIMUSensor(dai::IMUSensor::GYROSCOPE_CALIBRATED, strtol(argv[6],NULL,10));
    // it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
    imu->setBatchReportThreshold(1);
    // maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
    // if lower or equal to batchReportThreshold then the sending is always blocking on device
    // useful to reduce device's CPU load and number of lost packets, if CPU load is high on device side due to multiple nodes
    imu->setMaxBatchReports(10);
    // imu->enableFirmwareUpdate(true) //if i want to update

    // Linking
    monoLeft->out.link(depth->left);
    depth->rectifiedLeft.link(featureTrackerLeft->inputImage);
    if (allow == 1) {
        featureTrackerLeft->passthroughInputImage.link(xoutPassthroughFrameLeft->input);
    }
    featureTrackerLeft->outputFeatures.link(xoutTrackedFeaturesLeft->input);

    monoRight->out.link(depth->right);
    depth->rectifiedRight.link(featureTrackerRight->inputImage);
    if (allow == 1) {
        featureTrackerRight->passthroughInputImage.link(xoutPassthroughFrameRight->input);
    }
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

    dai::CalibrationHandler calibData = device.readCalibration2(); // read current calibration data
    double f, cx, cy;
    float baseline = calibData.getBaselineDistance(dai::CameraBoardSocket::CAM_B, dai::CameraBoardSocket::CAM_C, false) * 0.01f;
    calc_rect_cam_intri_extri(calibData, &f, &cx, &cy);
    float hfov = 2 * atanf(CAM_W / (2 * f));
    float vfov = 2 * atanf(CAM_H / (2 * f));
    // baseline equals to baseline amount in getBaselineDistance(bool=false)
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

    // verbose logging
    //device.setLogOutputLevel(dai::LogLevel::DEBUG);
    //device.setLogLevel(dai::LogLevel::DEBUG);

    // Output queues used to receive the results
    // 3rd argument when false specifies that old messages are overwritten when the queue is full
    auto outputFeaturesLeftQueue = device.getOutputQueue("trackedFeaturesLeft", 1, false); // size of queue, increased slightly to reduce jitter
    auto outputFeaturesRightQueue = device.getOutputQueue("trackedFeaturesRight", 1, false);
    if (allow == 1) {
        auto passthroughImageLeftQueue = device.getOutputQueue("passthroughFrameLeft", 1, false);
        auto passthroughImageRightQueue = device.getOutputQueue("passthroughFrameRight", 1, false);
    }
    auto disp_queue = device.getOutputQueue("disparity", 1, false);
    auto imuQueue = device.getOutputQueue("imu", 5, false);
    // Input queue to send runtime FeatureTracker config updates (optional)
    auto inputFeatureTrackerConfigQueue = device.getInputQueue("trackedFeaturesConfig");

    // sequence numbers initialisation
    int l_seq = -1, r_seq = -2, disp_seq = -3;

    // tools for variable processing
    std::vector<std::uint8_t> disp_frame; // disparity frame data
    uint16_t* pDisp_frame16 = nullptr; // pointer to transdormed disparity frame data
    std::vector<dai::TrackedFeature> l_features, r_features; // vectors containing features
    std::unordered_map<int, dai::Point2f> l_prv_features, r_prv_features; // vectors containing features from previous frame
    std::unordered_map<int, dai::Point2f> r_cur_features; // right image current features indexed map
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::steady_clock::duration> features_tp, prv_features_tp; // timestamps of frames
    std::unordered_map<int, int> lr_id_mapping; // features detected in left image paired to features in right
    // Mats for visualization CURRENTLY UNAVAILABLE BECAUSE OF MISSING DEPENDENCY
    // cv::Mat leftFrame, rightFrame;

    // Clear queue events
    //jakaskerl suggest remove this line
    //https://discuss.luxonis.com/d/3484-getqueueevent-takes-much-additional-time/7
    //device.getQueueEvents();

    float l_sum = 0.0, r_sum = 0.0, disp_sum = 0.0;
    int l_count = 0, r_count = 0, disp_count = 0;

    while(camera_run) {
        // Micro-optimizations (kept behavior the same):
        // 1) Cache a single host timestamp per loop to avoid calling steady_clock::now() multiple times.
        // 2) Reserve the right-features map before inserting to avoid rehashing.
        // 3) Use reinterpret_cast for the disparity pointer to make intent explicit.
        // Note: tracked-features remain copied from the packet to preserve original semantics.
        auto q_name = device.getQueueEvent();
        auto now_host = std::chrono::steady_clock::now();

        if (q_name == "trackedFeaturesLeft") { // waits until specified queue gets a message
            auto data = outputFeaturesLeftQueue->get<dai::TrackedFeatures>();
            l_features = data->trackedFeatures;
            l_seq = data->getSequenceNum(); // retrieve sequence number
            features_tp = data->getTimestampDevice(); // timestamp from camera
            l_sum += std::chrono::duration<float, std::milli>(now_host - data->getTimestamp()).count();
            ++l_count;
        } else if (q_name == "trackedFeaturesRight") {
            auto data = outputFeaturesRightQueue->get<dai::TrackedFeatures>();
            r_features = data->trackedFeatures;
            r_seq = data->getSequenceNum();
            r_sum += std::chrono::duration<float, std::milli>(now_host - data->getTimestamp()).count();
            ++r_count;
            // Reserve capacity to avoid repeated rehashing when filling the map.
            r_cur_features.clear();
            r_cur_features.reserve(r_features.size());
            for (const auto &feature : r_features) {
                r_cur_features[feature.id] = feature.position; // map features to indexes
            }
        } else if (q_name == "disparity") {
            auto disp_data = disp_queue->get<dai::ImgFrame>();
            disp_seq = disp_data->getSequenceNum();
            disp_frame = disp_data->getData(); // return only disparity data from frame
            // Use reinterpret_cast to express that we're reinterpreting raw bytes as uint16_t samples.
            pDisp_frame16 = reinterpret_cast<uint16_t*>(disp_frame.data());
            disp_sum += std::chrono::duration<float, std::milli>(now_host - disp_data->getTimestamp()).count();
            ++disp_count;
        } else if (q_name == "imu") {
            // IMU branch optimizations (safe, behavior-preserving):
            // - use const refs to avoid accidental copies
            // - reuse a single small buffer per IMU-event (stack-allocated once per q_name)
            // - cache timestamp conversion into a local double to avoid repeated expression overhead
            // - explicitly ignore sendto return value to avoid compiler warnings
            auto imuData = imuQueue->get<dai::IMUData>();
            const auto &imuPackets = imuData->packets;

            // Small stack buffer reused for every packet in this batch (keeps one allocation, tiny and fast)
            double imu_buf[7];
            for (const auto &imuPacket : imuPackets) {
                const auto &acc = imuPacket.acceleroMeter;
                const auto &gyro = imuPacket.gyroscope;

                // compute timestamp once per packet
                double ts = std::chrono::duration<double>(gyro.getTimestampDevice().time_since_epoch()).count();
                imu_buf[0] = ts;

                // translate to ROS frame (axis remap used elsewhere in project)
                imu_buf[1] = -acc.z;
                imu_buf[2] = -acc.y;
                imu_buf[3] = -acc.x;
                imu_buf[4] = -gyro.z;
                imu_buf[5] = -gyro.y;
                imu_buf[6] = -gyro.x;

                // send immediately; ignore return value intentionally (non-blocking local socket typical)
                (void)sendto(ipc_sock, imu_buf, sizeof(imu_buf), 0, (const struct sockaddr*)&imu_addr, sizeof(struct sockaddr_un));
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
            // prepare a local buffer: [count, timestamp, feature blocks...]
            std::vector<double> features_msg(2 + NUMBEROF_DATA * MAXIMUM_FEATURES);
            features_msg[1] = std::chrono::duration<double>(features_tp.time_since_epoch()).count(); // duration between epoch and timestamp of frame expressed in seconds
            size_t buf_index = 2; // first two slots are occupied with timestamps
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
                        features_msg[buf_index + 0] = static_cast<double>(l_feature.id); // store id
                        features_msg[buf_index + 1] = cur_un_x; // store normalised position x
                        features_msg[buf_index + 2] = cur_un_y; // and y
                        features_msg[buf_index + 3] = x; // store position x
                        features_msg[buf_index + 4] = y; // and y
                        features_msg[buf_index + 5] = vx; // store x 
                        features_msg[buf_index + 6] = vy; // and y speed
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
                        features_msg[buf_index + 7] = cur_un_x;
                        features_msg[buf_index + 8] = cur_un_y;
                        features_msg[buf_index + 9] = x;
                        features_msg[buf_index + 10] = y;
                        features_msg[buf_index + 11] = vx;
                        features_msg[buf_index + 12] = vy;

                        if (c < MAXIMUM_FEATURES) { // maximum number of features
                            ++c;
                            buf_index += NUMBEROF_DATA; // move to next position in buffer accordingly
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
                // Defensive: ensure disparity buffer is valid before indexing
                float disp = 0.0f;
                size_t disp_idx = static_cast<size_t>(row) * CAM_W + static_cast<size_t>(col);
                size_t disp_len = 0;
                if (!disp_frame.empty()) disp_len = disp_frame.size() / sizeof(uint16_t);
                if (pDisp_frame16 == nullptr || disp_idx >= disp_len) {
                    // No valid disparity for this pixel -> skip
                    continue;
                }
                disp = static_cast<float>(pDisp_frame16[disp_idx]) / 8.0f; // disparity value at pixel position
                if (disp > 0) { // if there exists a disparity
                    for (const auto &r_feature : r_features) {
                        float dy = y - r_feature.position.y; // difference between l and accredited to noise
                        float dx = x - disp - r_feature.position.x; // difference = noise and also disparity (epipolar shift)
                        if (dy * dy + dx * dx <= PAIR_DIST_SQ) { //pair found, aim for 95 percentile?
                            lr_id_mapping[l_feature.id] = r_feature.id; // persists over multiple frames if feature is repeatedly found
                            // same as left side
                            double dt = std::chrono::duration<double>(features_tp - prv_features_tp).count();
                            if (dt <= 1e-9) dt = 1e-6; // guard against division by zero / tiny dt
                            double vx = 0, vy = 0;
                            auto prv_pos = l_prv_features.find(l_feature.id);
                            if (prv_pos != l_prv_features.end()) {
                                vx = (cur_un_x - prv_pos->second.x) / dt;
                                vy = (cur_un_y - prv_pos->second.y) / dt;
                            }
                            features_msg[buf_index + 0] = static_cast<double>(l_feature.id);
                            features_msg[buf_index + 1] = cur_un_x;
                            features_msg[buf_index + 2] = cur_un_y;
                            features_msg[buf_index + 3] = x;
                            features_msg[buf_index + 4] = y;
                            features_msg[buf_index + 5] = vx;
                            features_msg[buf_index + 6] = vy;

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
                            features_msg[buf_index + 7] = cur_un_x;
                            features_msg[buf_index + 8] = cur_un_y;
                            features_msg[buf_index + 9] = x;
                            features_msg[buf_index + 10] = y;
                            features_msg[buf_index + 11] = vx;
                            features_msg[buf_index + 12] = vy;

                            if (c < MAXIMUM_FEATURES) {
                                ++c;
                                buf_index += NUMBEROF_DATA;
                            }

                            break;
                        }
                    }
                }
            }
            // logging every 60 frames
            num_frames++;
            if (num_frames > 60) {
                num_frames = 0;
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
            if (c < MIN_FEATURES) std::cout << "WARNING: too few features: " << c << "\n";
            // sending features
            if (c > 0) {
                if (c > 0 && imu_ok) {
                    features_msg[0] = static_cast<double>(c);
                    ssize_t sent = sendto(ipc_sock, features_msg.data(), static_cast<size_t>(2 + NUMBEROF_DATA * c) * sizeof(double), 0, (struct sockaddr*)&features_addr, sizeof(struct sockaddr_un));
                    // if (sent == -1) {
                    //     perror("features data send failed");
                    //     camera_run = 0;
                    // }
                }
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
        }
    }

    close(ipc_sock);
    std::cout << "bye\n";

    return 0;
}