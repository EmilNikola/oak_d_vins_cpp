/* 
   CONFIG: kamery jsou nastaveny na neexistujici rozliseni 480, protoze pro 9282 existuje jen 800,720 a 400p
   ???   : proc pouzivam stereoRectify od openCV misto stereoDepth z DepthAI (mozna proc pouzivam oboji), nenabizi depthai
           vycet pozadovanych parametru?
   LINUX : jaka presne yaml file je otevirana pri spousteni tohoto programu -- to budu resit az na Linuxu
   CONFIG: current fps je 20 myslim ze to jde vytahnout k 30, target features also overit
   CONFIG/??: proc croppujeme vystup obraz?
   CONFIG: projit funkce a nastaveni Stereodepth nastaveni, to je asi to hlavni
   CONFIG/??: imu data are extracted raw
   CONFIG: report rate na senzorech porovnat s datasheetovymi specifikacemi, take rate jak jsou data odesilana nejak probrat

   DepthAI: configure, track features, create disparity map, handle IMU
   OpenCV : rectify using OAK-D data
   wh
*/
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>

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
#include <sys/ipc.h> // inter process communication
#include <sys/shm.h> // shared process facility

// computer vision
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

// ROS
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/image.hpp"

// Includes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"
#include "deque"
#include "unordered_map"
#include "unordered_set"

using namespace std::chrono_literals;

#define MAX_FEATURES_COUNT 60
//#define H264_STREAMING        komprese videa
#define VIDEO_FPS 20
#define VIDEO_BITRATE 1500
#define DEPTH_SUBPIXEL // rozdilnost je pocitana subpixelovou aproximaci, vhodne na dalku

struct MyPoint2d {
    double x = 0;
    double y = 0;
    MyPoint2d() {}
    MyPoint2d(double px, double py) {
        x = px;
        y = py;
    }
};

double big_buf[14*MAX_FEATURES_COUNT+2]; // 2 spots reserved for timestamps

// ROS image publisher thread
sensor_msgs::msg::Image mono_img;
sensor_msgs::msg::Image disp_img;
bool mono_img_avail = false;
bool disp_img_avail = false;
bool img_pub_go = true;

void img_pub_func(rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mono_img_pub, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr disp_img_pub) {
    while (img_pub_go) {
        if (mono_img_avail) {
            mono_img_pub->publish(mono_img); // access Publisher functions through pointer
            mono_img_avail = false;
        }
        if (disp_img_avail) {
            disp_img_pub->publish(disp_img);
            disp_img_avail = false;
        }
        std::this_thread::sleep_for(10ms);
    }
}

void calc_rect_cam_intri(dai::CalibrationHandler calibData, double* f, double* cx, double* cy, int cam_w, int cam_h) {
    std::cout << "stereo baseline:" << calibData.getBaselineDistance(dai::CameraBoardSocket::CAM_B, dai::CameraBoardSocket::CAM_C, false) << " cm\n CAMERA EXTRINSICS:\n";
    auto imu_ext = calibData.getCameraToImuExtrinsics(dai::CameraBoardSocket::CAM_B, true);
    for (auto& row : imu_ext) {
        for (float val: row) {
            printf("extrinsics param: %f ", val);
        }
        printf("\n");
    }

    auto l_intrinsics = calibData.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B, cam_w, cam_h);
    float data[9];
    int i = -1;
    for (auto row : l_intrinsics) {
        for (auto val : row) {
            data[++i] = val;
        }
    }
    cv::Mat l_m = cv::Mat(3, 3, CV_32FC1, data); // 3x3 matrix of instrinsics as 32bit float

    auto r_intrinsics = calibData.getCameraIntrinsics(dai::CameraBoardSocket::CAM_C, cam_w, cam_h);
    i = -1;
    for (auto row : r_intrinsics) {
        for (auto val : row) {
            data[++i] = val;
        }
    }
    cv::Mat r_m = cv::Mat(3, 3, CV_32FC1, data); // 3x3 matrix of instrinsics as 32bit float
    std:cout << "camera intrinsics left\n" << l_m << "\n right \n" << r_m << "\n";

    auto l_d = calibData.getDistortionCoefficients(dai::CameraBoardSocket::CAM_B);
    auto r_d = calibData.getDistortionCoefficients(dai::CameraBoardSocket::CAM_C);
    auto extrinsics = calibData.getCameraExtrinsics(dai::CameraBoardSocket::CAM_B, dai::CameraBoardSocket::CAM_C);

    cv::Mat r = (cv::Mat_<double>(3,3) << extrinsics[0][0], extrinsics[0][1], extrinsics[0][2], extrinsics[1][0], extrinsics[1][1], extrinsics[1][2], extrinsics[2][0], extrinsics[2][1], extrinsics[2][2]);
    cv::Mat t = (cv::Mat_<double>(3,1) << extrinsics[0][3], extrinsics[1][3], extrinsics[2][3]);
    std::cout << "stereo extrinsics\n" << r << "\n" << t << "\n";
    cv::Mat r1, r2, p1, p2, q;
    cv::stereoRectify(l_m, l_d, r_m, r_d, cv::Size(cam_w, cam_h), r, t, r1, r2, p1, p2, q, cv::CALIB_ZERO_DISPARITY, 0); // rectification transforms for stereo images alignment

    std::cout << "P1\n" << p1 << "\nP2\n" << p2 << "\n"; // projection matrices in rectified coordinates sysem for both cameras

    *f = p1.at<double>(0, 0); // rectified focal length
    *cx = p1.at<double>(0, 2); // horizontal center for rectified camera
    *cy = p1.at<double>(1, 2); // vertical center for rectified camera
}

int main(int argc, char **argv) {
    bool h264_ok = false;
    int cam_w, cam_h;
    bool imu_ok = false;
    int ccc=0;
    int long_ms=0;
    int short_ms=INT_MAX;
    unsigned int mono_pub_c = 0;
    unsigned int disp_pub_c = 0;
    unsigned char seq_num = 0;

    if (argc < 2) {
        printf("usage: %s imu_tk_cali.yml\n", argv[0]);
        return 0;
    }

    rclcpp::init(argc, argv); // initialize ROS2 client lib
    auto ros_node = rclcpp::Node::make_shared("feature_tracker"); // create node
    // create message publishers, only one mono camera suffices
    auto mono_img_pub = ros_node->create_publisher<sensor_msgs::msg::Image>("mono_left", rclcpp::QoS(1).best_effort().durability_volatile());
    auto disp_img_pub = ros_node->create_publisher<sensor_msgs::msg::Image>("disparity", rclcpp::QoS(1).best_effort().durability_volatile());

#ifdef REC_IMU
    FILE* imu_file = fopen("oakd_imu.bin", "w");
    FILE* features_file = fopen("oakd_features.bin", "w");
#endif

#ifdef H264_STREAMING
    // IPC shared memory
    key_t key = ftok("shmfile", 65);
    // shmget returns an identifier in shmid
    int shmid = shmget(key, 500000, 0666 | IPC_CREAT);
    unsigned char* h264_pkt_data = (unsigned char*)shmat(shmid, (void*)0, 0);
#endif

    cv::FileStorage imu_yml;

    // load accelerometer calibration data - reads YAML nodes and extracts them into matrix variables
    imu_yml.open(argv[1], cv::FileStorage::READ);
    cv::Mat acc_mis_align, acc_scale, acc_bias;
    imu_yml["acc_misalign"] >> acc_mis_align;
    imu_yml["acc_scale"] >> acc_scale;
    imu_yml["acc_bias"] >> acc_bias;
    cv::Mat acc_cor = acc_mis_align * acc_scale;
    std::cout << "Accelerometer calibration data" << acc_mis_align<<"\n"<<acc_scale<<"\n"<<acc_bias<<"\n";
    // load gyroscope calibration data - reads YAML nodes and extracts them into matrix variables
    cv::Mat gyro_mis_align, gyro_scale, gyro_bias;
    imu_yml["gyro_misalign"] >> gyro_mis_align;
    imu_yml["gyro_scale"] >> gyro_scale;
    imu_yml["gyro_bias"] >> gyro_bias;
    cv::Mat gyro_cor = gyro_mis_align * gyro_scale;
    imu_yml.release();
    std::cout<< "Gyroscope calibration data" <<gyro_mis_align<<"\n"<<gyro_scale<<"\n"<<gyro_bias<<"\n";

    // creating unix sockets, ipc_local_addr to receive and other two to send data
    struct sockaddr_un ipc_local_addr, imu_addr, features_addr;
    memset(&ipc_local_addr, 0, sizeof(struct sockaddr_un));
    ipc_local_addr.sun_family = AF_UNIX;
    strcpy(ipc_local_addr.sun_path, "/tmp/chobits_2222");
    if ((int ipc_sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
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
    auto featureTrackerLeft = pipeline.create<dai::node::FeatureTracker>();
    auto featureTrackerRight = pipeline.create<dai::node::FeatureTracker>();
    auto imu = pipeline.create<dai::node::IMU>();
    auto depth = pipeline.create<dai::node::StereoDepth>();
#ifdef H264_STREAMING
    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    auto videoEnc = pipeline.create<dai::node::VideoEncoder>();
#endif
    // note to transform images
    auto manip = pipeline.create<dai::node::ImageManip>();

    // xlink nodes to link the other ones together
    auto xoutTrackedFeaturesLeft = pipeline.create<dai::node::XLinkOut>();
    auto xoutTrackedFeaturesRight = pipeline.create<dai::node::XLinkOut>();
    auto xout_disp = pipeline.create<dai::node::XLinkOut>();
    auto xout_imu = pipeline.create<dai::node::XLinkOut>();
    auto xout_mono = pipeline.create<dai::node::XLinkOut>();
#ifdef H264_STREAMING
    auto xout_h264 = pipeline.create<dai::node::XLinkOut>();
#endif

    // specify some stream names over which nodes receive their data
    xoutTrackedFeaturesLeft->setStreamName("trackedFeaturesLeft");
    xoutTrackedFeaturesRight->setStreamName("trackedFeaturesRight");
    xout_disp->setStreamName("disparity");
    xout_imu->setStreamName("imu");
    xout_mono->setStreamName("mono");
#ifdef H264_STREAMING
    xout_h264->setStreamName("h264");
#endif

    // Properties
    // monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);  pro 9282 tento resolution neexistuje
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_800_P);
    monoLeft->setCamera("left");
    monoLeft->setFps(20);
    // monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_800_P);
    monoRight->setCamera("right");
    monoRight->setFps(20);

    manip->initialConfig.setCropRect(0.2, 0.2, 0.8, 0.8);

    featureTrackerLeft->initialConfig.setNumTargetFeatures(16*5);
    featureTrackerRight->initialConfig.setNumTargetFeatures(16*5);
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
    depth->initialConfig.setMedianFilter(dai::MedianFilter::MEDIAN_OFF);
    //depth->initialConfig.setConfidenceThreshold(0); // maximum confidence that it holds a valid value
    depth->setLeftRightCheck(true);
    depth->setExtendedDisparity(false);
#ifdef DEPTH_SUBPIXEL
    depth->setSubpixel(true);
    depth->setSubpixelFractionalBits(3);
#else
    depth->setSubpixel(false);
#endif
    depth->setDepthAlign(dai::RawStereoDepthConfig::AlgorithmControl::DepthAlign::RECTIFIED_LEFT);
    depth->setAlphaScaling(0);

    imu->enableIMUSensor(dai::IMUSensor::ACCELEROMETER_RAW, 200);
    imu->enableIMUSensor(dai::IMUSensor::GYROSCOPE_RAW, 200);
    // it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
    // above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
    imu->setBatchReportThreshold(1);
    // maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
    // if lower or equal to batchReportThreshold then the sending is always blocking on device
    // useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
    imu->setMaxBatchReports(10);

#ifdef H264_STREAMING
    camRgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setFps(VIDEO_FPS);
    camRgb->setNumFramesPool(2, 2, 2, 2, 2);
    videoEnc->setDefaultProfilePreset(VIDEO_FPS, dai::VideoEncoderProperties::Profile::H264_MAIN);
    videoEnc->setKeyframeFrequency(VIDEO_FPS*2);
    videoEnc->setBitrateKbps(VIDEO_BITRATE);
    videoEnc->setNumFramesPool(2);
    videoEnc->input.setQueueSize(2);
    videoEnc->input.setBlocking(false);
#endif

    // Linking
    monoLeft->out.link(depth->left);
    depth->rectifiedLeft.link(featureTrackerLeft->inputImage);
    featureTrackerLeft->outputFeatures.link(xoutTrackedFeaturesLeft->input);

    monoRight->out.link(depth->right);
    depth->rectifiedRight.link(featureTrackerRight->inputImage);
    featureTrackerRight->outputFeatures.link(xoutTrackedFeaturesRight->input);

    depth->disparity.link(xout_disp->input);
    imu->out.link(xout_imu->input);
    monoLeft->out.link(manip->inputImage);
    manip->out.link(xout_mono->input);
#ifdef H264_STREAMING
    //monoLeft->out.link(videoEnc->input);
    camRgb->video.link(videoEnc->input);
    videoEnc->bitstream.link(xout_h264->input);
#endif

    // Connect to device and start pipeline
    dai::Device device(pipeline);

    // vycitani parametru ze zarizeni
    std::cout << "Usb speed: " << device.getUsbSpeed() << "\n";
    std::cout << "Device name: " << device.getDeviceName() << " Product name: " << device.getProductName() << "\n";
    if (device.getDeviceName() != "OAK-D-PRO") printf("not OAK-D-PRO\n");

    cam_w = monoLeft->getResolutionWidth();
    cam_h = monoLeft->getResolutionHeight();
    printf("stereo res %dx%d\n", cam_w, cam_h);

    dai::CalibrationHandler calibData = device.readCalibration2();
    double f, cx, cy;
    float baseline = calibData.getBaselineDistance(dai::CameraBoardSocket::CAM_B, dai::CameraBoardSocket::CAM_C, false) * 0.01f;
    calc_rect_cam_intri(calibData, &f, &cx, &cy, cam_w, cam_h);
    float hfov = 2 * atanf(cam_w / (2 * f));
    float vfov = 2 * atanf(cam_h / (2 * f));
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
        std::cout << "(TEMPORARY PRINTOUT: stereo pair baseline:" << s_pair.baseline << " cm\n";
    }

    // verbose logging 
    //device.setLogOutputLevel(dai::LogLevel::DEBUG);
    //device.setLogLevel(dai::LogLevel::DEBUG);

    // Output queues used to receive the results
    // 3. argument when false specifies that old messages are overwritten when the queue is full
    auto outputFeaturesLeftQueue = device.getOutputQueue("trackedFeaturesLeft", 1, false);
    auto outputFeaturesRightQueue = device.getOutputQueue("trackedFeaturesRight", 1, false);
    auto disp_queue = device.getOutputQueue("disparity", 1, false);
    auto imuQueue = device.getOutputQueue("imu", 10, false);
    auto mono_queue = device.getOutputQueue("mono", 1, false);
#ifdef H264_STREAMING
    auto video = device.getOutputQueue("h264", 1, false);
#endif

    // sequence numbers initialisation
    int l_seq = -1, r_seq = -2, disp_seq = -3;


#ifdef DEPTH_SUBPIXEL
    uint16_t* disp_data;
#else
    uint8_t* disp_data;
#endif
    // vectors for both cameras containing features
    std::vector<dai::TrackedFeature> l_features, r_features;
    // mapping 2D points to an integer
    std::map<int, MyPoint2d> l_prv_features, r_prv_features;
    double features_ts, prv_features_ts;
    double latest_exp_t = 0; // latest exposure time as seconds
    //double last_acc_t = 0;

    // a time point constructed from steady clock equal to steady clock duration
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::steady_clock::duration> l_ft_tp;

    // Clear queue events
    //device.getQueueEvents();

    // create and run thread with img_pub_func
    std::thread img_pub_worker(img_pub_func, mono_img_pub, disp_img_pub);

    while(rclcpp::ok()) {
        auto q_name = device.getQueueEvent();

        // LOADING QUEUE MESSAGES ----------------------------------------------------------------
        if (q_name == "trackedFeaturesLeft") { // waits until specified queue gets a message
            auto data = outputFeaturesLeftQueue->get<dai::TrackedFeatures>();
            l_features = data->trackedFeatures;
            l_seq = data->getSequenceNum(); // retrieve sequence number
            features_ts = std::chrono::duration<double>(data->getTimestampDevice().time_since_epoch()).count(); // takes timepoint duration since clock epoch and returns it as seconds via count
            l_ft_tp = data->getTimestamp(); // image timestamp relative to clock "now"
            //std::cout << "l ft " << l_seq << " latency:" << std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - features_tp).count() << " ms\n";
        } else if (q_name == "trackedFeaturesRight") {
            auto data = outputFeaturesRightQueue->get<dai::TrackedFeatures>();
            r_features = data->trackedFeatures;
            r_seq = data->getSequenceNum();
            //std::cout << "r ft " << r_seq << " latency:" << std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - data->getTimestamp()).count() << " ms\n";
        } else if (q_name == "disparity") {
            auto disp_frame = disp_queue->get<dai::ImgFrame>();
            disp_seq = disp_frame->getSequenceNum();
            auto disp_frame_data = disp_frame->getData(); // return only disparity data from frame
#ifdef DEPTH_SUBPIXEL
            disp_data = (uint16_t*)disp_frame_data.data(); // point at the first disparity value in frame
#else
            disp_data = (uint8_t*)disp_frame_data.data();
#endif
        /
            latest_exp_t = std::chrono::duration<double>(disp_frame->getExposureTime()).count();
            //std::cout << "stereo " << disp_seq << " latency:" << std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - disp_data->getTimestamp()).count() << " ms\n";
            
            // publish disparity every four frames in queue via ROS
            disp_pub_c++;
            if (disp_pub_c > 3) {
                disp_pub_c = 0;
                disp_img.header.stamp = ros_node->get_clock()->now();
                disp_img.height = disp_frame->getHeight();
                disp_img.width = disp_frame->getWidth();
                disp_img.is_bigendian = 0;
#ifdef DEPTH_SUBPIXEL
                disp_img.encoding = "mono16";
                disp_img.step = disp_img.width*2;
#else
                disp_img.encoding = "mono8";
                disp_img.step = disp_img.width;
#endif
                disp_img.data = disp_frame_data;
                disp_img_avail = true;
            }
        } else if (q_name == "imu") {
            auto imuData = imuQueue->get<dai::IMUData>();
            auto imuPackets = imuData->packets;
            for(const auto& imuPacket : imuPackets) {
                auto& acc = imuPacket.acceleroMeter;
                auto& gyro = imuPacket.gyroscope;
                //std::cout << "imu latency, acc:" << std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - acc.getTimestamp()).count() << " ms, gyro:" << std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - gyro.getTimestamp()).count() << " ms\n";
                big_buf[0] = std::chrono::duration<double>(acc.getTimestampDevice().time_since_epoch()).count();
                //if (big_buf[0] - last_acc_t > 0.007) printf("imu jitter %f\n", big_buf[0] - last_acc_t);
                //last_acc_t = big_buf[0];
                cv::Mat acc_raw = (cv::Mat_<double>(3,1) << acc.x, acc.y, acc.z);
                cv::Mat1d acc_cali = acc_cor * (acc_raw - acc_bias);
                cv::Mat gyro_raw = (cv::Mat_<double>(3,1) << gyro.x, gyro.y, gyro.z);
                cv::Mat1d gyro_cali = gyro_cor * (gyro_raw - gyro_bias);
                // translate to ros frame, easier to understand in rviz
                big_buf[1] = -acc_cali(2,0);
                big_buf[2] = -acc_cali(0,0);
                big_buf[3] = acc_cali(1,0);
                big_buf[4] = -gyro_cali(2,0);
                big_buf[5] = -gyro_cali(0,0);
                big_buf[6] = gyro_cali(1,0);
                sendto(ipc_sock, big_buf, 7*sizeof(double), 0, (struct sockaddr*)&imu_addr, sizeof(struct sockaddr_un));
            }
            if (!imu_ok) {
                imu_ok = true;
                std::cout<< "imu ok\n";
            }
        } else if (q_name == "h264") {
#ifdef H264_STREAMING
            if (!h264_ok) {
                h264_ok = true;
                std::cout<<"h264 ok\n";
            }
            auto h264Packet = video->get<dai::ImgFrame>();
            auto h264data = h264Packet->getData();
            //
            // IPC data structure: length+data+seq_num+flag
            // length - size of h264 bitstream (4 bytes)
            // data - h264 bitstream (n bytes)
            // seq_num - 1 byte
            // flag - 1 byte
            //
            int h264_pkt_len = h264data.size();
			//printf("h264enc len=%d\n", h264Packet->getData().size());
			h264_pkt_data[0] = (h264_pkt_len >> 24) & 0xff;
			h264_pkt_data[1] = (h264_pkt_len >> 16) & 0xff;
			h264_pkt_data[2] = (h264_pkt_len >> 8) & 0xff;
			h264_pkt_data[3] = h264_pkt_len & 0xff;
			memcpy(h264_pkt_data+4, h264data.data(), h264_pkt_len);
            h264_pkt_data[h264_pkt_len+4] = seq_num;
            seq_num++;
            if(seq_num == 0xff) seq_num = 0;
            h264_pkt_data[h264_pkt_len+5] = 1;
#endif
        } else if (q_name == "mono") {
            auto img_frame = mono_queue->get<dai::ImgFrame>();
            mono_pub_c++;
            if (mono_pub_c > 3) {
                mono_pub_c = 0;
                mono_img.header.stamp = ros_node->get_clock()->now();
                mono_img.height = img_frame->getHeight();
                mono_img.width = img_frame->getWidth();
                mono_img.is_bigendian = 0;
                mono_img.encoding = "mono8";
                mono_img.step = mono_img.width;
                mono_img.data = img_frame->getData();
                mono_img_avail = true;
            }
        }

        if (l_seq == r_seq && r_seq == disp_seq) { // executes if left, right and disparity frames align
            //auto t1 = std::chrono::steady_clock::now();
            l_seq = -1;
            r_seq = -2;
            disp_seq = -3;
            std::map<int , MyPoint2d> features;
            int c = 0;
            features_ts = features_ts - latest_exp_t * 0.5; // move timestamp to middle of exposition
            big_buf[1] = features_ts; // timestamp occupies second place in the buffer
            double* buf_ptr = big_buf + 2; // skip 2 occupied spots
            for (const auto &l_feature : l_features) {
                float x = l_feature.position.x;
                float y = l_feature.position.y;
                double cur_un_x = l_inv_k11 * x + l_inv_k13; // normalised x position
                double cur_un_y = l_inv_k22 * y + l_inv_k23; // normalised y position
                features[l_feature.id] = MyPoint2d(cur_un_x, cur_un_y);
                // convert float to int
                int row = y;
                int col = x;
                // round up float to int
                int ceil_row = ceilf(y);
                int ceil_col = ceilf(x);
#ifdef DEPTH_SUBPIXEL
                float disps[4] = {0};
                float disp;
                // if conditions are true, four neigboring pixels are stored, used for further processing
                disps[0] = disp_data[row * cam_w + col] / 8.0f; // set first sample as center pixel, divided by 8 because set to 3 fractional bits 2^3=8
                if (ceil_row != (int)y && ceil_row < cam_h) disps[1] = disp_data[ceil_row * cam_w + col] / 8.0f; // neighbor if rounding result differs
                if (ceil_col != (int)x && ceil_col < cam_w) disps[2] = disp_data[row * cam_w + ceil_col] / 8.0f; // same but for witdth instead of height
                if (disps[1] && disps[2]) { // only created if the previous two are valid, meaning new value is created
                    disps[3] = disp_data[ceil_row * cam_w + ceil_col] / 8.0f;
                }
#else
                int disps[4] = {0};
                int disp;
                disps[0] = disp_data[row * cam_w + col];
                if (ceil_row != (int)y && ceil_row < cam_h) disps[1] = disp_data[ceil_row * cam_w + col];
                if (ceil_col != (int)x && ceil_col < cam_w) disps[2] = disp_data[row * cam_w + ceil_col];
                if (disps[1] && disps[2]) {
                    disps[3] = disp_data[ceil_row * cam_w + ceil_col];
                }
#endif
                // disps neigbor pixels processing
                for (int i = 0; i < 4; i++) {
                    disp = disps[i];
                    if (disp > 0) {
                        bool pair_found = false;
                        for (const auto &r_feature : r_features) {
                            float dy = y - r_feature.position.y; // calculating difference between images
                            float dx = x - disp - r_feature.position.x; // calculating difference, horizontal is affected by disparity
                            if (fabsf(dy) <= 1 && fabsf(dx) <= 2) { //pair found, only one is necessary, thresholds chosen empirically
                                pair_found = true;
                                double dt = features_ts - prv_features_ts; // timestamp difference between current and latest dispartity frame
                                double vx = 0, vy = 0;
                                auto prv_pos = l_prv_features.find(l_feature.id); // check if the same feature was present in previous frame, returns iterator
                                if (prv_pos != l_prv_features.end()) { // checks if previous line found instance
                                    vx = (cur_un_x - prv_pos->second.x) / dt; // normalised speed (prv_pos->second.x is x value of the map of previous frame)
                                    vy = (cur_un_y - prv_pos->second.y) / dt;
                                }
                                buf_ptr[0] = l_feature.id; // store id
                                buf_ptr[1] = cur_un_x; // store normalised position x
                                buf_ptr[2] = cur_un_y; // and y
                                buf_ptr[3] = x; // store position x
                                buf_ptr[4] = y; // and y
                                buf_ptr[5] = vx; // store x 
                                buf_ptr[6] = vy; // and y speed

                                x = r_feature.position.x; // storing right feature positions instead of left
                                y = r_feature.position.y;
                                vx = 0; // resetting speed
                                vy = 0;
                                cur_un_x = r_inv_k11 * x + r_inv_k13;
                                cur_un_y = r_inv_k22 * y + r_inv_k23;
                                prv_pos = r_prv_features.find(r_feature.id);
                                if (prv_pos != r_prv_features.end()) {
                                    vx = (cur_un_x - prv_pos->second.x) / dt;
                                    vy = (cur_un_y - prv_pos->second.y) / dt;
                                }
                                buf_ptr[7] = cur_un_x; // continuing to fill the buffer with right features
                                buf_ptr[8] = cur_un_y;
                                buf_ptr[9] = x;
                                buf_ptr[10] = y;
                                buf_ptr[11] = vx;
                                buf_ptr[12] = vy;
                                buf_ptr[13] = f * baseline / disp; // compute depth for the pixel

                                if (c < MAX_FEATURES_COUNT) { // maximum of 60 features is stored from one image
                                    ++c;
                                    buf_ptr += 14; // continue saving the next feature
                                }

                                break;
                            }
                        }
                        if (pair_found) break; // no need to iterate for every 'disp' if match is found
                    }
                }
            }
            int cost_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - l_ft_tp).count(); // difference between now and last timestamp
            // updating minimum and maximum latency
            if (cost_ms > long_ms) long_ms = cost_ms;
            if (cost_ms < short_ms) short_ms = cost_ms;
            // logging every 60 frames
            ccc++;
            if (ccc > 60) { 
                ccc = 0;
                std::cout << l_features.size() << " features " << c << " LR matched, latency(ms) max " << long_ms << ", min " << short_ms  << "\n";
                // refresh latencies for next group
                long_ms = 0;
                short_ms = INT_MAX;
                //latency ~ 40 ms
            }
            if (c < 10) printf("WARNING: too few features: %d\n", c);
            // sending features
            if (imu_ok && c > 0) {
                big_buf[0] = c;
                sendto(ipc_sock, big_buf, 14*sizeof(double)*c+2*sizeof(double), 0, (struct sockaddr*)&features_addr, sizeof(struct sockaddr_un));
            }
            // moving frame to make place for new one
            l_prv_features = features;
            prv_features_ts = features_ts;
            r_prv_features.clear();
            for (const auto &r_feature : r_features) {
                r_prv_features[r_feature.id] = MyPoint2d(r_inv_k11 * r_feature.position.x + r_inv_k13, r_inv_k22 * r_feature.position.y + r_inv_k23); // load with normalised position of image that is previous for the next iteration
            }
            //auto t2 = std::chrono::steady_clock::now();
            //std::cout << std::chrono::duration<float, std::milli>(t2-t1).count() << " ms\n";
        }
    }

    close(ipc_sock);

#ifdef REC_IMU
    fclose(imu_file);
    fclose(features_file);
#endif

#ifdef H264_STREAMING
    shmdt(h264_pkt_data);
    shmctl(shmid, IPC_RMID, NULL);
#endif

    img_pub_go = false;
    // block thread main until this thread finishes execution
    img_pub_worker.join();

    rclcpp::shutdown();
    printf("bye\n");

    return 0;
}