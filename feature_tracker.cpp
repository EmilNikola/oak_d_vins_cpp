/*
odstranit potencialni deleni nulou (rychlosti)
je pDisp_frame16 safe? k zamysleni

3a algoritmy*

*/

#include <iostream>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <sys/types.h> // data types for working with processes
#include <sys/socket.h> // communication endpoints
#include <unistd.h> // posix api
#include <string.h>
#include <string>
#include <sys/un.h> // unix sockets
#include <signal.h> // signal handling
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>

// computer vision
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <memory>
#include <fstream>
// threading / IPC helpers (removed - sender thread disabled)

// Includes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"
#include <unordered_map>

#include <atomic>

// Pairing distance threshold (squared pixel distance). Default: 9 (3px radius).
// Can be overridden at runtime via argv[25].
static int pair_dist_sq = 9;
#define MIN_FEATURES 10
#define MAXIMUM_FEATURES 118
#define NUMBEROF_DATA 13

// camera parameters as specified by THE_400_P
static int CAM_W = 640;
static int CAM_H = 400;


static constexpr const char* FRAME_MAGIC = "VFRM";
static constexpr std::uint8_t FRAME_SIDE_LEFT = 0;
static constexpr std::uint8_t FRAME_SIDE_RIGHT = 1;

static void sendFramePacket(int inet_sock, const struct sockaddr_in& inet_addr_remote, const cv::Mat& gray, std::uint8_t side) {
    std::vector<std::uint8_t> jpg;
    cv::imencode(".jpg", gray, jpg, {cv::IMWRITE_JPEG_QUALITY, 60});
    if(jpg.empty()) return;

    std::vector<std::uint8_t> pkt(5 + jpg.size());
    std::memcpy(pkt.data(), FRAME_MAGIC, 4);
    pkt[4] = side;
    std::memcpy(pkt.data() + 5, jpg.data(), jpg.size());

    sendto(inet_sock, pkt.data(), pkt.size(), 0, (struct sockaddr*)&inet_addr_remote, sizeof(inet_addr_remote));
}

volatile sig_atomic_t camera_run = 1;
void sig_func(int sig) {
    camera_run = 0;
}

void calc_rect_cam_intri_extri(dai::CalibrationHandler calibData, double* f, double* cx, double* cy) {
    
    float data[9]; // left and right intrinsics

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

// Helper function to calculate mean and standard deviation
struct Statistics {
    float mean;
    float stddev;
    float min_val;
    float max_val;
    size_t count;
};

Statistics calculateStats(const std::vector<float>& data) {
    Statistics stats = {0.0f, 0.0f, 0.0f, 0.0f, data.size()};
    if (data.empty()) return stats;
    
    // Calculate mean
    float sum = 0.0f;
    stats.min_val = data[0];
    stats.max_val = data[0];
    for (float val : data) {
        sum += val;
        stats.min_val = std::min(stats.min_val, val);
        stats.max_val = std::max(stats.max_val, val);
    }
    stats.mean = sum / data.size();
    
    // Calculate standard deviation
    float var_sum = 0.0f;
    for (float val : data) {
        var_sum += (val - stats.mean) * (val - stats.mean);
    }
    stats.stddev = std::sqrt(var_sum / data.size());
    
    return stats;
}

int main(int argc, char **argv) {
    bool imu_ok = false;
    int num_frames=0; // number of frames
    int allow = 0;
    if(argc > 10) {
        allow = static_cast<int>(strtol(argv[10], NULL, 10));
    }

    int depth_preset_idx = 0; // default
    if(argc > 4) {
        depth_preset_idx = static_cast<int>(strtol(argv[4], NULL, 10));
    }

    int median_idx = -1;
    int ext_disp_idx = -1;
    int subpixel_flag = -1;
    int subpixel_frac = -1;
    int motion_idx = -1; // 0 = Lucas-Kanade (default), 1 = HW motion estimation
    if(argc > 11) median_idx = static_cast<int>(strtol(argv[11], NULL, 10));
    if(argc > 12) ext_disp_idx = static_cast<int>(strtol(argv[12], NULL, 10));
    if(argc > 13) subpixel_frac = static_cast<int>(strtol(argv[13], NULL, 10));
    if(argc > 14) subpixel_flag = static_cast<int>(strtol(argv[14], NULL, 10));
    if(argc > 15) motion_idx = static_cast<int>(strtol(argv[15], NULL, 10));

    // Optional pairing distance override: argv[25]
    // argv[24] is used for features_log_filename; so use argv[25] for pair distance.
    if(argc > 25) {
        int tmp = static_cast<int>(strtol(argv[25], NULL, 10));
        if(tmp > 0) {
            pair_dist_sq = tmp;
        }
    }

    // Feature tracker thresholds: argv[26] = tracking error, argv[27] = Harris score
    int tracking_error_threshold = 50000; // default
    int harris_score_threshold = 200000; // default
    if(argc > 26) {
        int tmp = static_cast<int>(strtol(argv[26], NULL, 10));
        if(tmp > 0) {
            tracking_error_threshold = tmp;
        }
    }
    if(argc > 27) {
        int tmp = static_cast<int>(strtol(argv[27], NULL, 10));
        if(tmp > 0) {
            harris_score_threshold = tmp;
        }
    }

    // StereoDepth confidence threshold: argv[28] (range 0-255, default 200)
    int depth_confidence_threshold = 200; // default
    if(argc > 28) {
        int tmp = static_cast<int>(strtol(argv[28], NULL, 10));
        if(tmp >= 0 && tmp <= 255) {
            depth_confidence_threshold = tmp;
        }
    }

    CAM_W = static_cast<int>(strtol(argv[18], NULL, 10));
    CAM_H = static_cast<int>(strtol(argv[19], NULL, 10));

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

    // NOTE: sender thread removed — sending will use direct sendto() again.
    memset(&imu_addr, 0, sizeof(struct sockaddr_un));
    imu_addr.sun_family = AF_UNIX;
    strcpy(imu_addr.sun_path, "/tmp/chobits_imu");

    memset(&features_addr, 0, sizeof(struct sockaddr_un));
    features_addr.sun_family = AF_UNIX;
    strcpy(features_addr.sun_path, "/tmp/chobits_features");

    // Optional INET (UDP) socket to send features over network to remote visualiser ! wrong argvs, allow flags
    bool inet_enabled = false;
    int inet_sock = -1;
    struct sockaddr_in inet_addr_remote;
    // parse inet_enable flag if provided
    int inet_flag = 0;
    if(argc > 21) {
        inet_flag = static_cast<int>(strtol(argv[21], NULL, 10));
    }
    if(inet_flag) {
        if(argc > 23 && argv[22] && argv[23]) {
            const char* remote_host = argv[22];
            int remote_port = static_cast<int>(strtol(argv[23], NULL, 10));
            inet_sock = socket(AF_INET, SOCK_DGRAM, 0);
            if(inet_sock == -1) {
                perror("inet socket creation failed");
            } else {
                // set non-blocking to avoid main-loop stalls
                int flags = fcntl(inet_sock, F_GETFL, 0);
                if(flags != -1) fcntl(inet_sock, F_SETFL, flags | O_NONBLOCK);

                memset(&inet_addr_remote, 0, sizeof(inet_addr_remote));
                inet_addr_remote.sin_family = AF_INET;
                inet_addr_remote.sin_port = htons(remote_port);
                if(inet_pton(AF_INET, remote_host, &inet_addr_remote.sin_addr) != 1) {
                    std::cerr << "Invalid remote host IP: " << remote_host << "\n";
                    close(inet_sock);
                    inet_sock = -1;
                } else {
                    inet_enabled = true;
                }
            }
        } else {
            std::cerr << "inet_enable set but remote_host/remote_port missing (argv[22],argv[23]) - disabling inet.\n";
        }
    }

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
    // Map argv[20] (resolution selection) to DepthAI enum; default to THE_400_P
    dai::MonoCameraProperties::SensorResolution res = dai::MonoCameraProperties::SensorResolution::THE_400_P;
    if(argc > 20 && argv[20]) {
        std::string r(argv[20]);
        if(r == "THE_400_P") res = dai::MonoCameraProperties::SensorResolution::THE_400_P;
        else if(r == "THE_720_P") res = dai::MonoCameraProperties::SensorResolution::THE_720_P;
        else if(r == "THE_800_P") res = dai::MonoCameraProperties::SensorResolution::THE_800_P;
        // add other enum string mappings as needed
    }
    monoLeft->setResolution(res);
    monoLeft->setCamera("left");
    monoLeft->setFps(strtol(argv[3],NULL,10));
    monoRight->setResolution(res);
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
    //featureMaintainer now configurable via command line arguments
    featureTrackerConfig.featureMaintainer.minimumDistanceBetweenFeatures = 50;
    featureTrackerConfig.featureMaintainer.lostFeatureErrorThreshold = tracking_error_threshold;
    featureTrackerConfig.featureMaintainer.trackedFeatureThreshold = harris_score_threshold;
    

    featureTrackerLeft->initialConfig.set(featureTrackerConfig);
    featureTrackerRight->initialConfig.set(featureTrackerConfig);
    printConfig("after", featureTrackerConfig);

    // according to API refrence for both Shaves and Memory slices, maximum number is allocated
    // hardware resources hint: number of shaves (VPU cores) and memory slices (CMX partitions)
    // default values (safe): 2 shaves, 2 slices
    int numShaves = 2;
    int numSlices = 2;
    if(argc > 16) numShaves = static_cast<int>(strtol(argv[16], NULL, 10));
    if(argc > 17) numSlices = static_cast<int>(strtol(argv[17], NULL, 10));
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
                depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_3x3);
                break;
            case 2:
                depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_5x5);
                break;
            case 3:
                depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_7x7);
                break;
            default:
                depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_3x3);
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

    // Set confidence threshold for depth computation (0-255)
    depth->initialConfig.setConfidenceThreshold(depth_confidence_threshold);

    // Compute disparity divisor from fractional bits used by StereoDepth.
    // Device encodes disparity as fixed-point: raw = disparity_pixels * (2^fractional_bits).
    // Default historically used in this code was 3 fractional bits -> divisor = 8.
    float disparity_divisor = 8.0f;
    if(subpixel_flag <= 0) {
        // Subpixel disabled -> integer disparities
        disparity_divisor = 1.0f;
    } else {
        if(subpixel_frac >= 0) {
            // Use explicit fractional bits provided by CLI
            disparity_divisor = static_cast<float>(1u << subpixel_frac);
        } else {
            // Subpixel enabled but fractional bits not provided; keep historical default
            disparity_divisor = 8.0f;
        }
    }
    std::cout << "disparity_divisor=" << disparity_divisor << " (subpixel_flag=" << subpixel_flag << ", subpixel_frac=" << subpixel_frac << ")\n";

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
    // Declare passthrough queues in outer scope so main loop can access them when `allow==1`.
    decltype(outputFeaturesLeftQueue) passthroughImageLeftQueue = nullptr;
    decltype(outputFeaturesRightQueue) passthroughImageRightQueue = nullptr;
    if (allow == 1) {
        passthroughImageLeftQueue = device.getOutputQueue("passthroughFrameLeft", 1, false);
        passthroughImageRightQueue = device.getOutputQueue("passthroughFrameRight", 1, false);
    }
    auto disp_queue = device.getOutputQueue("disparity", 1, false);
    auto imuQueue = device.getOutputQueue("imu", 5, false);
    // Input queue to send runtime FeatureTracker config updates (optional)
    auto inputFeatureTrackerConfigQueue = device.getInputQueue("trackedFeaturesConfig");

    // Optional tracked features logging to CSV. Provide filename via argv[24],
    // otherwise defaults to "tracked_features.csv" in the current directory.
    std::string features_log_filename = "tracked_features.csv";
    if(argc > 24 && argv[24]) features_log_filename = std::string(argv[24]);
    std::ofstream features_log(features_log_filename, std::ios::out | std::ios::trunc);
    if(features_log.is_open()) {
        features_log << "device_time,side,seq,id,age,x,y,harrisScore,trackingError\n";
    } else {
        std::cerr << "Warning: failed to open features log file: " << features_log_filename << "\n";
    }
    size_t features_log_counter = 0;

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

    // Reserve unordered_map buckets upfront to avoid rehashing during the hot loop.
    // Use a small safety factor to reduce rehash frequency.
    // Allow overriding the compile-time MAXIMUM_FEATURES via argv[2] (numMaxFeatures).
    int max_features = MAXIMUM_FEATURES;
    if(argc > 2) max_features = static_cast<int>(strtol(argv[2], NULL, 10));
    const size_t reserve_size = static_cast<size_t>(max_features) * 2;
    l_prv_features.reserve(reserve_size);
    r_prv_features.reserve(reserve_size);
    r_cur_features.reserve(reserve_size);
    lr_id_mapping.reserve(reserve_size);
    // Mats for visualization CURRENTLY UNAVAILABLE BECAUSE OF MISSING DEPENDENCY
    // cv::Mat leftFrame, rightFrame;

    // Preallocate features message buffer once and reuse to avoid per-frame heap allocations.
    std::vector<double> features_msg;
    features_msg.resize(2 + NUMBEROF_DATA * max_features);

    // (spatial grid removed) pairing currently uses direct scan; keep simple for now

    // Statistics tracking for parameter optimization
    std::vector<float> stat_pair_distances;      // actual pairing distances (sqrt of pair_dist_sq)
    std::vector<float> stat_tracking_errors;     // actual tracking errors
    std::vector<float> stat_harris_scores;       // actual harris scores
    std::vector<float> stat_disparities;         // actual disparities for confidence analysis
    stat_pair_distances.reserve(10000);
    stat_tracking_errors.reserve(10000);
    stat_harris_scores.reserve(10000);
    stat_disparities.reserve(10000);

    // Clear queue events
    //jakaskerl suggest remove this line
    //https://discuss.luxonis.com/d/3484-getqueueevent-takes-much-additional-time/7
    //device.getQueueEvents();

    float l_sum = 0.0, r_sum = 0.0, disp_sum = 0.0;
    int l_count = 0, r_count = 0, disp_count = 0;
    // Timing instrumentation (ms)
    using clock = std::chrono::steady_clock;
    std::chrono::duration<double, std::milli> t_tracked_left{0}, t_tracked_right{0}, t_disp{0}, t_imu{0}, t_pairing{0}, t_send{0}, t_prev_update{0};
    uint64_t cnt_tracked_left = 0, cnt_tracked_right = 0, cnt_disp = 0, cnt_imu = 0, cnt_pairing = 0, cnt_send = 0, cnt_prev_update = 0;

    while(camera_run) {
        // Micro-optimizations (kept behavior the same):
        // 1) Cache a single host timestamp per loop to avoid calling steady_clock::now() multiple times.
        // 2) Reserve the right-features map before inserting to avoid rehashing.
        // 3) Use reinterpret_cast for the disparity pointer to make intent explicit.
        // Note: tracked-features remain copied from the packet to preserve original semantics.
        auto q_name = device.getQueueEvent();
        auto now_host = std::chrono::steady_clock::now();
        
        // Handle passthrough frames (optional). If passthrough is enabled and we receive
        // a passthroughFrame event, forward the JPEG over the INET socket if configured.
        if (q_name == "passthroughFrameLeft") {
            if(passthroughImageLeftQueue) {
                auto inPassthroughFrameLeft = passthroughImageLeftQueue->get<dai::ImgFrame>();
                if(inet_enabled && inet_sock != -1) {
                    auto dataLeft = inPassthroughFrameLeft->getData();
                    int hLeft = inPassthroughFrameLeft->getHeight();
                    int wLeft = inPassthroughFrameLeft->getWidth();
                    cv::Mat frameLeft(hLeft, wLeft, CV_8UC1, (void*)dataLeft.data());
                    sendFramePacket(inet_sock, inet_addr_remote, frameLeft, FRAME_SIDE_LEFT);
                }
            }
            continue;
        } else if (q_name == "passthroughFrameRight") {
            if(passthroughImageRightQueue) {
                auto inPassthroughFrameRight = passthroughImageRightQueue->get<dai::ImgFrame>();
                if(inet_enabled && inet_sock != -1) {
                    auto dataRight = inPassthroughFrameRight->getData();
                    int hRight = inPassthroughFrameRight->getHeight();
                    int wRight = inPassthroughFrameRight->getWidth();
                    cv::Mat frameRight(hRight, wRight, CV_8UC1, (void*)dataRight.data());
                    sendFramePacket(inet_sock, inet_addr_remote, frameRight, FRAME_SIDE_RIGHT);
                }
            }
            continue;
        }
        if (q_name == "trackedFeaturesLeft") { // waits until specified queue gets a message
            auto t0 = clock::now();
            auto data = outputFeaturesLeftQueue->get<dai::TrackedFeatures>();
            l_features = data->trackedFeatures;
            l_seq = data->getSequenceNum(); // retrieve sequence number
            features_tp = data->getTimestampDevice(); // timestamp from camera
            l_sum += std::chrono::duration<float, std::milli>(now_host - data->getTimestamp()).count();
            ++l_count;
            t_tracked_left += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(clock::now() - t0);
            ++cnt_tracked_left;
        } else if (q_name == "trackedFeaturesRight") {
            auto t0 = clock::now();
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
            t_tracked_right += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(clock::now() - t0);
            ++cnt_tracked_right;
        } else if (q_name == "disparity") {
            auto t0 = clock::now();
            auto disp_data = disp_queue->get<dai::ImgFrame>();
            disp_seq = disp_data->getSequenceNum();
            disp_frame = disp_data->getData(); // return only disparity data from frame
            // Use reinterpret_cast to express that we're reinterpreting raw bytes as uint16_t samples.
            pDisp_frame16 = reinterpret_cast<uint16_t*>(disp_frame.data());
            disp_sum += std::chrono::duration<float, std::milli>(now_host - disp_data->getTimestamp()).count();
            ++disp_count;
            t_disp += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(clock::now() - t0);
            ++cnt_disp;
        } else if (q_name == "imu") {
            auto t0 = clock::now();
            auto imuData = imuQueue->get<dai::IMUData>();
            const auto imuPackets = imuData->packets;
                    for(const auto& imuPacket : imuPackets) {
                        const auto& acc = imuPacket.acceleroMeter;
                        const auto& gyro = imuPacket.gyroscope;
                        // prepare local buffer for IMU message
                        double imu_buf[7];
                        imu_buf[0] = std::chrono::duration<double>(gyro.getTimestampDevice().time_since_epoch()).count();
                        // translate to ros frame
                        imu_buf[1] = -acc.z;
                        imu_buf[2] = -acc.y;
                        imu_buf[3] = -acc.x;
                        imu_buf[4] = -gyro.z;
                        imu_buf[5] = -gyro.y;
                        imu_buf[6] = -gyro.x;
                        // synchronous send and measure
                        auto t1 = clock::now();
                        ssize_t sent = sendto(ipc_sock, imu_buf, sizeof(imu_buf), 0, (struct sockaddr*)&imu_addr, sizeof(struct sockaddr_un));
                        t_send += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(clock::now() - t1);
                        ++cnt_send;
                        // if (sent == -1) {
                        //     perror("imu data send failed");
                        //     camera_run = 0;
                        // }
                    }
            t_imu += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(clock::now() - t0);
            ++cnt_imu;
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
            // Reuse preallocated features_msg to eliminate per-frame allocations.
            // The buffer capacity is MAXIMUM_FEATURES; we'll only send the first
            // (2 + NUMBEROF_DATA * c) entries when enqueuing the message.
            features_msg[1] = std::chrono::duration<double>(features_tp.time_since_epoch()).count(); // frame timestamp (seconds)
            size_t buf_index = 2; // first two slots are occupied with timestamps
            auto t_pair0 = clock::now();
            for (const auto &l_feature : l_features) {
                if (c >= max_features) break;
                // Collect statistics
                stat_harris_scores.push_back(static_cast<float>(l_feature.harrisScore));
                stat_tracking_errors.push_back(static_cast<float>(l_feature.trackingError));
                
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
                        if (dt <= 1e-9) dt = 1e-6; // guard against division by zero / tiny dt
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

                        if (c < max_features) { // maximum number of features
                            ++c;
                            // Log paired features to CSV if enabled
                            if (features_log.is_open()) {
                                double ts = std::chrono::duration<double>(features_tp.time_since_epoch()).count();
                                features_log << ts << ",L," << l_seq << "," << l_feature.id << "," << l_feature.age << "," << l_feature.position.x << "," << l_feature.position.y << "," << l_feature.harrisScore << "," << l_feature.trackingError << "\n";
                                ++features_log_counter;
                            }
                            buf_index += NUMBEROF_DATA; // move to next position in buffer accordingly
                        }

                        continue;
                    }
                }
                // rounding down
                int col = roundf(x);
                int row = roundf(y);
                // setting bounds for possible values
                if (col < 0) col = 0;
                if (col > CAM_W - 1) col = CAM_W - 1;
                if (row < 0) row = 0;
                if (row > CAM_H - 1) row = CAM_H - 1;
                // Defensive: ensure disparity buffer is valid before indexing
                float disp = 0.0f;
                size_t disp_idx = static_cast<size_t>(row) * CAM_W + static_cast<size_t>(col);

                if (subpixel_flag <= 0) {
                    // 8-bit: one byte per pixel
                    size_t disp_len = disp_frame.size();
                    if (disp_idx >= disp_len) continue;
                    disp = static_cast<float>(disp_frame[disp_idx]);
                } else {
                    // 16-bit subpixel: two bytes per pixel
                    size_t disp_len = disp_frame.size() / sizeof(uint16_t);
                    if (pDisp_frame16 == nullptr || disp_idx >= disp_len) continue;
                    disp = static_cast<float>(pDisp_frame16[disp_idx]) / disparity_divisor;
                }

                if (disp > 0) { // if there exists a disparity
                    for (const auto &r_feature : r_features) {
                        float dy = y - r_feature.position.y; // difference between l and accredited to noise
                        float dx = x - disp - r_feature.position.x; // difference = noise and also disparity (epipolar shift)
                        float pair_dist = dy * dy + dx * dx;
                        if (pair_dist <= pair_dist_sq) { //pair found, aim for 95 percentile?
                            // Collect statistics for this pair
                            stat_pair_distances.push_back(std::sqrt(pair_dist));
                            stat_disparities.push_back(disp);
                            
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

                            if (c < max_features) {
                                ++c;
                                // Log paired features to CSV if enabled
                                if (features_log.is_open()) {
                                    double ts = std::chrono::duration<double>(features_tp.time_since_epoch()).count();
                                    features_log << ts << ",L," << l_seq << "," << l_feature.id << "," << l_feature.age << "," << l_feature.position.x << "," << l_feature.position.y << "," << l_feature.harrisScore << "," << l_feature.trackingError << "\n";
                                    ++features_log_counter;
                                }
                                buf_index += NUMBEROF_DATA;
                            }

                            break;
                        }
                    }
                }
            }
            
            // pairing duration
            t_pairing += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(clock::now() - t_pair0);
            ++cnt_pairing;
            // logging every 60 frames
            num_frames++;
            if (num_frames > 60) {
                num_frames = 0;
                std::cout << c << " features\n";
                std::cout << "average latency LEFT: " << l_sum/l_count << " ms\n";
                std::cout << "average latency RIGHT: " << r_sum/r_count << " ms\n";
                std::cout << "average latency DISPARITY: " << disp_sum/disp_count << " ms\n";
                // Print timing instrumentation averages (ms)
                if (cnt_tracked_left) std::cout << "avg trackedLeft: " << (t_tracked_left.count() / static_cast<double>(cnt_tracked_left)) << " ms over " << cnt_tracked_left << " samples\n";
                if (cnt_tracked_right) std::cout << "avg trackedRight: " << (t_tracked_right.count() / static_cast<double>(cnt_tracked_right)) << " ms over " << cnt_tracked_right << " samples\n";
                if (cnt_disp) std::cout << "avg disparity: " << (t_disp.count() / static_cast<double>(cnt_disp)) << " ms over " << cnt_disp << " samples\n";
                if (cnt_imu) std::cout << "avg imu: " << (t_imu.count() / static_cast<double>(cnt_imu)) << " ms over " << cnt_imu << " samples\n";
                if (cnt_pairing) std::cout << "avg pairing: " << (t_pairing.count() / static_cast<double>(cnt_pairing)) << " ms over " << cnt_pairing << " samples\n";
                if (cnt_prev_update) std::cout << "avg prev_update: " << (t_prev_update.count() / static_cast<double>(cnt_prev_update)) << " ms over " << cnt_prev_update << " samples\n";
                if (cnt_send) std::cout << "avg send: " << (t_send.count() / static_cast<double>(cnt_send)) << " ms over " << cnt_send << " samples\n";
                l_sum = 0.0;
                r_sum = 0.0;
                disp_sum = 0.0;
                l_count = 0;
                r_count = 0;
                disp_count = 0;
                // reset instrumentation counters and timers
                t_tracked_left = decltype(t_tracked_left){0};
                t_tracked_right = decltype(t_tracked_right){0};
                t_disp = decltype(t_disp){0};
                t_imu = decltype(t_imu){0};
                t_pairing = decltype(t_pairing){0};
                t_prev_update = decltype(t_prev_update){0};
                t_send = decltype(t_send){0};
                cnt_tracked_left = cnt_tracked_right = cnt_disp = cnt_imu = cnt_pairing = cnt_prev_update = cnt_send = 0;
            }
            if (c < MIN_FEATURES) std::cout << "WARNING: too few features: " << c << "\n";
            size_t featurePayloadBytes = static_cast<size_t>(2 + NUMBEROF_DATA * c) * sizeof(double);
            // sending features
                if (c > 0 && imu_ok) {
                    features_msg[0] = static_cast<double>(c);
                    // Send features message directly (synchronous) and measure send time
                    auto t1 = clock::now();
                    ssize_t sent = sendto(ipc_sock, features_msg.data(), featurePayloadBytes, 0, (struct sockaddr*)&features_addr, sizeof(struct sockaddr_un));
                    t_send += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(clock::now() - t1);
                    ++cnt_send;
                    // if (sent == -1) {
                    //     perror("features data send failed");
                    //     camera_run = 0;
                    // }
            }
            // Also optionally send features over INET (UDP). Socket is non-blocking; if send would block
            // we drop the packet to avoid stalling the main loop.
            if(inet_enabled && inet_sock != -1) {
                features_msg[0] = static_cast<double>(c);
                ssize_t sent2 = sendto(inet_sock, features_msg.data(), featurePayloadBytes, 0, (struct sockaddr*)&inet_addr_remote, sizeof(inet_addr_remote));
                if (sent2 == -1) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        // socket busy, drop packet (non-fatal)
                    } else {
                        // log unexpected errors once
                        perror("inet sendto failed");
                    }
                }
            }

            // current frame moved to previous to make place for new frame
            {
                auto t0 = clock::now();
                l_prv_features = features;
                prv_features_tp = features_tp;
                r_prv_features.clear();
                for (const auto &r_feature : r_features) {
                    r_prv_features[r_feature.id] = dai::Point2f(
                        static_cast<float>(r_inv_k11 * r_feature.position.x + r_inv_k13),
                        static_cast<float>(r_inv_k22 * r_feature.position.y + r_inv_k23)
                    );
                }
                t_prev_update += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(clock::now() - t0);
                ++cnt_prev_update;
            }
        }
    }

    close(ipc_sock);
    if(inet_sock != -1) close(inet_sock);
    if(features_log.is_open()) {
        features_log.flush();
        features_log.close();
        std::cout << "Wrote " << features_log_counter << " feature records to " << features_log_filename << "\n";
    }
    
    // Print parameter statistics for optimization
    std::cout << "\n=== PARAMETER OPTIMIZATION STATISTICS ===\n";
    std::cout << "Use these values with ±1σ to tune your parameters\n\n";
    
    auto pair_dist_stats = calculateStats(stat_pair_distances);
    std::cout << "argv[25] - Pair Distance (pixels):\n";
    std::cout << "  Mean: " << pair_dist_stats.mean << " σ=" << pair_dist_stats.stddev << "\n";
    std::cout << "  Range: [" << pair_dist_stats.min_val << ", " << pair_dist_stats.max_val << "]\n";
    std::cout << "  Suggested: " << static_cast<int>(pair_dist_stats.mean) 
              << " (mean), " << static_cast<int>(pair_dist_stats.mean + pair_dist_stats.stddev) 
              << " (mean+σ), " << static_cast<int>(pair_dist_stats.mean + 2*pair_dist_stats.stddev) << " (mean+2σ)\n";
    std::cout << "  Currently set to (squared): " << pair_dist_sq << " (= " 
              << std::sqrt(pair_dist_sq) << " pixels)\n\n";
    
    auto tracking_error_stats = calculateStats(stat_tracking_errors);
    std::cout << "argv[26] - Tracking Error Threshold:\n";
    std::cout << "  Mean: " << tracking_error_stats.mean << " σ=" << tracking_error_stats.stddev << "\n";
    std::cout << "  Range: [" << tracking_error_stats.min_val << ", " << tracking_error_stats.max_val << "]\n";
    std::cout << "  Suggested: " << static_cast<int>(tracking_error_stats.mean) 
              << " (mean), " << static_cast<int>(tracking_error_stats.mean + tracking_error_stats.stddev) 
              << " (mean+σ), " << static_cast<int>(tracking_error_stats.mean + 2*tracking_error_stats.stddev) << " (mean+2σ)\n";
    std::cout << "  Currently set to: " << tracking_error_threshold << "\n\n";
    
    auto harris_score_stats = calculateStats(stat_harris_scores);
    std::cout << "argv[27] - Harris Score Threshold:\n";
    std::cout << "  Mean: " << harris_score_stats.mean << " σ=" << harris_score_stats.stddev << "\n";
    std::cout << "  Range: [" << harris_score_stats.min_val << ", " << harris_score_stats.max_val << "]\n";
    std::cout << "  Suggested: " << static_cast<int>(harris_score_stats.mean) 
              << " (mean), " << static_cast<int>(harris_score_stats.mean + harris_score_stats.stddev) 
              << " (mean+σ), " << static_cast<int>(harris_score_stats.mean + 2*harris_score_stats.stddev) << " (mean+2σ)\n";
    std::cout << "  Currently set to: " << harris_score_threshold << "\n\n";
    
    auto disparity_stats = calculateStats(stat_disparities);
    std::cout << "argv[28] - Disparity Quality (for confidence threshold):\n";
    std::cout << "  Mean disparity: " << disparity_stats.mean << " σ=" << disparity_stats.stddev << "\n";
    std::cout << "  Range: [" << disparity_stats.min_val << ", " << disparity_stats.max_val << "]\n";
    std::cout << "  Note: Higher confidence threshold (closer to 255) filters noisy disparities\n";
    std::cout << "  Currently set to: " << depth_confidence_threshold << "\n";
    std::cout << "  Suggestion: Try values from 150-250 based on depth quality needs\n\n";
    
    std::cout << "Total paired features analyzed: " << pair_dist_stats.count << "\n";
    std::cout << "================================\n";
    
    std::cout << "bye\n";

    return 0;
}