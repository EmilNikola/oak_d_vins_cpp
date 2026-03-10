#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <cmath>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <poll.h>

// Includes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"
#include "deque"
#include "unordered_map"
#include "unordered_set"

// line color
static const auto lineColor = cv::Scalar(200, 0, 200);
// point color
static const auto pointColor = cv::Scalar(0, 0, 255);

// main class used
class FeatureTrackerDrawer {
   private:
    // point size in pixels
    static const int circleRadius = 2;
    // longest a path can get in pixels
    static const int maxTrackedFeaturesPathLength = 30;
    // for how many frames the feature is tracked   !!
    static int trackedFeaturesPathLength;

    // alias for id=uint32_t
    using featureIdType = decltype(dai::TrackedFeature::id);

    // container of types featureIdType as hash-based set - search, insertion and removal have constant-time complexity
    std::unordered_set<featureIdType> trackedIDs;
    // container of featureIdType matched to 2D coordinates stored in double opened queue container - deque
    std::unordered_map<featureIdType, std::deque<dai::Point2f>> trackedFeaturesPath;
    // unimportant
    std::string trackbarName;
    std::string windowName;

   public:
    // function that takes vector of newly tracked features, and stores them
    void trackFeaturePath(std::vector<dai::TrackedFeature>& features) {
        std::unordered_set<featureIdType> newTrackedIDs;
        for(auto& currentFeature : features) {
            auto currentID = currentFeature.id;
            newTrackedIDs.insert(currentID);

            // if trackedFeaturesPath doesnt contain feature with currentID key, empty deque is inserted 
            //auto& path = trackedFeatures[currentID]; // can replace next 3 lines because of unordered set intrinsic duplicate prevention
            if(!trackedFeaturesPath.count(currentID)) {
                trackedFeaturesPath.insert({currentID, std::deque<dai::Point2f>()});
            }
            // takes a reference to either last feature or newly created empty deque - value at currentID
            std::deque<dai::Point2f>& path = trackedFeaturesPath.at(currentID);

            // adds x,y position to path at the end, if vector isnt big enough its automatically resized 
            path.push_back(currentFeature.position);
            // if size of path is greater than either one or amount set by trackedFeaturesPathLength, first element is removed
            while(path.size() > std::max<unsigned int>(1, trackedFeaturesPathLength)) {
                path.pop_front();
            }
        }
        
        // .count counts number of elements of that id, which can be either 1 or 0
        // if newTrrackedIDs doesnt contain ID of a feature in already tracked set,
        // it is then placed in featuresToRemove set - because it becomes useless
        std::unordered_set<featureIdType> featuresToRemove;
        for(auto& oldId : trackedIDs) {
            if(!newTrackedIDs.count(oldId)) {
                featuresToRemove.insert(oldId);
            }
        }
        // those features are then removed from trackedFeaturesPath
        for(auto& id : featuresToRemove) {
            trackedFeaturesPath.erase(id);
        }
        // currently processed features are moved back for the next iteration
        trackedIDs = newTrackedIDs;
    }

    // drawer skipped, requires to be sorted through for potentionally important info
    void drawFeatures(cv::Mat& img) {
        cv::setTrackbarPos(trackbarName.c_str(), windowName.c_str(), trackedFeaturesPathLength);

        for(auto& featurePath : trackedFeaturesPath) {
            std::deque<dai::Point2f>& path = featurePath.second;
            unsigned int j = 0;
            for(j = 0; j < path.size() - 1; j++) {
                auto src = cv::Point(path[j].x, path[j].y);
                auto dst = cv::Point(path[j + 1].x, path[j + 1].y);
                cv::line(img, src, dst, lineColor, 1, cv::LINE_AA, 0);
            }

            cv::circle(img, cv::Point(path[j].x, path[j].y), circleRadius, pointColor, -1, cv::LINE_AA, 0);
        }
    }

    // class constructor -- describe
    FeatureTrackerDrawer(std::string trackbarName, std::string windowName) : trackbarName(trackbarName), windowName(windowName) {
        cv::namedWindow(windowName.c_str());
        cv::createTrackbar(trackbarName.c_str(), windowName.c_str(), &trackedFeaturesPathLength, maxTrackedFeaturesPathLength, nullptr);
    }
};

// sets the amount of frames a feature is tracked across to 10
int FeatureTrackerDrawer::trackedFeaturesPathLength = 10;

int main(int argc, char** argv) {
    using namespace std;

    // Expect: visualiser <udp_port> [cam_w cam_h]
    int udp_port = 5005;
    int CAM_W = 640;
    int CAM_H = 400;
    if(argc > 1 && argv[1]) udp_port = std::atoi(argv[1]);
    if(argc > 2 && argv[2]) CAM_W = std::atoi(argv[2]);
    if(argc > 3 && argv[3]) CAM_H = std::atoi(argv[3]);

    std::cout << "Visualiser listening on UDP port " << udp_port << " (size " << CAM_W << "x" << CAM_H << ")\n";

    // Create drawers + windows
    const auto leftWindowName = "left";
    auto leftFeatureDrawer = FeatureTrackerDrawer("Feature tracking duration (frames)", leftWindowName);
    const auto rightWindowName = "right";
    auto rightFeatureDrawer = FeatureTrackerDrawer("Feature tracking duration (frames)", rightWindowName);

    // Create UDP socket and bind
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if(sock < 0) {
        perror("socket");
        return 1;
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(udp_port);
    if(bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sock);
        return 1;
    }

    const size_t BUF_SZ = 65536;
    std::vector<char> buf(BUF_SZ);

    while(true) {
        ssize_t n = recv(sock, buf.data(), buf.size(), 0);
        if(n <= 0) {
            if(n < 0) perror("recv");
            // wait a little then continue
            int key = cv::waitKey(1);
            if(key == 27) break;
            continue;
        }

        if(n % sizeof(double) != 0) {
            std::cerr << "Received packet length not multiple of double: " << n << "\n";
            continue;
        }
        size_t nd = n / sizeof(double);
        if(nd < 2) continue;

        std::vector<double> v(nd);
        std::memcpy(v.data(), buf.data(), n);

        int c = static_cast<int>(std::round(v[0]));
        if(c < 0) c = 0;
        if(2 + (size_t)c * NUMBEROF_DATA > nd) {
            std::cerr << "Packet claims " << c << " features but has only " << nd << " doubles\n";
            continue;
        }

        // Create blank frames (black)
        cv::Mat leftFrame(CAM_H, CAM_W, CV_8UC3, cv::Scalar(0,0,0));
        cv::Mat rightFrame(CAM_H, CAM_W, CV_8UC3, cv::Scalar(0,0,0));

        std::vector<dai::TrackedFeature> leftFeatures;
        std::vector<dai::TrackedFeature> rightFeatures;
        leftFeatures.reserve(c);
        rightFeatures.reserve(c);

        for(int i = 0; i < c; ++i) {
            size_t base = 2 + i * NUMBEROF_DATA;
            double id = v[base + 0];
            double lx = v[base + 3];
            double ly = v[base + 4];
            double rx = v[base + 9];
            double ry = v[base + 10];

            dai::TrackedFeature lf;
            lf.id = static_cast<decltype(lf.id)>(id);
            lf.position = dai::Point2f(static_cast<float>(lx), static_cast<float>(ly));
            leftFeatures.push_back(lf);

            dai::TrackedFeature rf;
            rf.id = static_cast<decltype(rf.id)>(id);
            rf.position = dai::Point2f(static_cast<float>(rx), static_cast<float>(ry));
            rightFeatures.push_back(rf);

            int lpx = static_cast<int>(std::round(lx));
            int lpy = static_cast<int>(std::round(ly));
            int rpx = static_cast<int>(std::round(rx));
            int rpy = static_cast<int>(std::round(ry));
            if(lpx >= 0 && lpx < CAM_W && lpy >= 0 && lpy < CAM_H) cv::circle(leftFrame, cv::Point(lpx, lpy), 3, cv::Scalar(0,0,255), -1);
            if(rpx >= 0 && rpx < CAM_W && rpy >= 0 && rpy < CAM_H) cv::circle(rightFrame, cv::Point(rpx, rpy), 3, cv::Scalar(0,0,255), -1);
        }

        if(!leftFeatures.empty()) {
            leftFeatureDrawer.trackFeaturePath(leftFeatures);
            leftFeatureDrawer.drawFeatures(leftFrame);
        }
        if(!rightFeatures.empty()) {
            rightFeatureDrawer.trackFeaturePath(rightFeatures);
            rightFeatureDrawer.drawFeatures(rightFrame);
        }

        cv::putText(leftFrame, "features: " + std::to_string(c), cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);
        cv::putText(rightFrame, "features: " + std::to_string(c), cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);

        cv::imshow(leftWindowName, leftFrame);
        cv::imshow(rightWindowName, rightFrame);
        int key = cv::waitKey(1);
        if(key == 27) break;
    }

    close(sock);
    return 0;
}