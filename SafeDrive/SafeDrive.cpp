///////////////////////////////////////////////////
// Written by: Abdulrahman Elgendy
// Last Update: 12-01-2022
//////////////////////////////////////////////////

#include "SafeDrive.h"

safeDrive::safeDrive(int deviceId)
{
    int NumEyeContourPts = 6;

    LeftEyeContour.reserve(NumEyeContourPts);
    RightEyeContour.reserve(NumEyeContourPts);
    for (int i = 0; i < NumEyeContourPts; i++) {
        LeftEyeContour.push_back(cv::Point(0, 0));
        RightEyeContour.push_back(cv::Point(0, 0));
    }

	cap.open(deviceId);
    if (!cap.isOpened()) {
        std::cout << "Video stream not operational, please check device ID" << std::endl;
    }
    else {
        std::cout << "Video stream setup and ready \n";
    }
    thresholdTrackBar();
}

void safeDrive::thresholdTrackBar()
{
    cv::namedWindow("Threshold Cutoff For Calibration", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Threshold", "Threshold Cutoff For Calibration", &ThresholdVal, ThresholdMax);
    cv::moveWindow("Threshold Cutoff For Calibration", 1150, 10);
}

cv::CascadeClassifier safeDrive::loadFaceClassifier()
{
    cv::CascadeClassifier faceClassifier;
    if (!faceClassifier.load("haarcascade_frontalface_default.xml")) {
        std::cout << "Face Classifier could not be loaded" << std::endl;
    }
    else {
        std::cout << "Face Classifier was loaded properly" << std::endl;
    }
    return faceClassifier;
}

std::vector<cv::Rect> safeDrive::detectFaces(cv::CascadeClassifier& faceClassifier, const cv::Mat& Frame)
{
    std::vector<cv::Rect> faceBoxes;
    faceClassifier.detectMultiScale(Frame, faceBoxes);
    return faceBoxes;
}

dlib::rectangle safeDrive::opencv_rect_to_dlib(cv::Rect rectangle)
{
    return dlib::rectangle((long)rectangle.tl().x, (long)rectangle.tl().y, (long)rectangle.br().x - 1, (long)rectangle.br().y - 1);
}

dlib::shape_predictor safeDrive::loadFaceLandmarkDetector() {
    dlib::shape_predictor sp;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    return sp;
}

safeDrive::EyeLandmarkPts safeDrive::extractEyeLandmarks(dlib::full_object_detection& FaceLandmarks) {
    cv::Point LSLeye = cv::Point(FaceLandmarks.part(36).x(), FaceLandmarks.part(36).y());
    cv::Point RSLeye = cv::Point(FaceLandmarks.part(39).x(), FaceLandmarks.part(39).y());
    cv::Point TLSLeye = cv::Point(FaceLandmarks.part(37).x(), FaceLandmarks.part(37).y());
    cv::Point TRSLeye = cv::Point(FaceLandmarks.part(38).x(), FaceLandmarks.part(38).y());
    cv::Point TMSLeye = cv::Point((FaceLandmarks.part(37).x() + FaceLandmarks.part(38).x()) / 2, (FaceLandmarks.part(37).y() + FaceLandmarks.part(38).y()) / 2);
    cv::Point BLSLeye = cv::Point(FaceLandmarks.part(41).x(), FaceLandmarks.part(41).y());
    cv::Point BRSLeye = cv::Point(FaceLandmarks.part(40).x(), FaceLandmarks.part(40).y());
    cv::Point BMSLeye = cv::Point((FaceLandmarks.part(41).x() + FaceLandmarks.part(40).x()) / 2, (FaceLandmarks.part(41).y() + FaceLandmarks.part(40).y()) / 2);

    cv::Point LSReye = cv::Point(FaceLandmarks.part(42).x(), FaceLandmarks.part(42).y());
    cv::Point RSReye = cv::Point(FaceLandmarks.part(45).x(), FaceLandmarks.part(45).y());
    cv::Point TLSReye = cv::Point(FaceLandmarks.part(43).x(), FaceLandmarks.part(43).y());
    cv::Point TRSReye = cv::Point(FaceLandmarks.part(44).x(), FaceLandmarks.part(44).y());
    cv::Point TMSReye = cv::Point((FaceLandmarks.part(43).x() + FaceLandmarks.part(44).x()) / 2, (FaceLandmarks.part(43).y() + FaceLandmarks.part(44).y()) / 2);
    cv::Point BLSReye = cv::Point(FaceLandmarks.part(47).x(), FaceLandmarks.part(47).y());
    cv::Point BRSReye = cv::Point(FaceLandmarks.part(46).x(), FaceLandmarks.part(46).y());
    cv::Point BMSReye = cv::Point((FaceLandmarks.part(47).x() + FaceLandmarks.part(46).x()) / 2, (FaceLandmarks.part(47).y() + FaceLandmarks.part(46).y()) / 2);

    LeftEyeContour.at(0) = LSLeye;
    LeftEyeContour.at(1) = TLSLeye;
    LeftEyeContour.at(2) = TRSLeye;
    LeftEyeContour.at(3) = RSLeye;
    LeftEyeContour.at(4) = BRSLeye;
    LeftEyeContour.at(5) = BLSLeye;

    RightEyeContour.at(0) = LSReye;
    RightEyeContour.at(1) = TLSReye;
    RightEyeContour.at(2) = TRSReye;
    RightEyeContour.at(3) = RSReye;
    RightEyeContour.at(4) = BRSReye;
    RightEyeContour.at(5) = BLSReye;

    EyeLandmarkPts EyePts;
    EyePts.ptsLeye = (const cv::Point*)cv::Mat(LeftEyeContour).data;
    EyePts.nptsLeye = cv::Mat(LeftEyeContour).rows;
    EyePts.ptsReye = (const cv::Point*)cv::Mat(RightEyeContour).data;
    EyePts.nptsReye = cv::Mat(RightEyeContour).rows;
    return EyePts;
}

safeDrive::EyeFrameSegments safeDrive::extractEyeSegmentsFromFrame(safeDrive::EyeLandmarkPts& EyePts, cv::Mat& FrameGrey)
{
    EyeFrameSegments EyeSegs;
    cv::Mat FullFrameEye;
    cv::Mat EyeMask = cv::Mat::zeros(FrameGrey.size(), FrameGrey.type());

    cv::fillPoly(EyeMask, &EyePts.ptsLeye, &EyePts.nptsLeye, 1, cv::Scalar(255, 255, 255));
    cv::fillPoly(EyeMask, &EyePts.ptsReye, &EyePts.nptsReye, 1, cv::Scalar(255, 255, 255));
    cv::bitwise_and(FrameGrey, EyeMask, FullFrameEye);

    const int MinEyeXLeye = std::min({ EyePts.ptsLeye[0].x, EyePts.ptsLeye[1].x, EyePts.ptsLeye[2].x, EyePts.ptsLeye[3].x, EyePts.ptsLeye[4].x, EyePts.ptsLeye[5].x });
    const int MaxEyeXLeye = std::max({ EyePts.ptsLeye[0].x, EyePts.ptsLeye[1].x, EyePts.ptsLeye[2].x, EyePts.ptsLeye[3].x, EyePts.ptsLeye[4].x, EyePts.ptsLeye[5].x });
    const int MinEyeYLeye = std::min({ EyePts.ptsLeye[0].y, EyePts.ptsLeye[1].y, EyePts.ptsLeye[2].y, EyePts.ptsLeye[3].y, EyePts.ptsLeye[4].y, EyePts.ptsLeye[5].y });
    const int MaxEyeYLeye = std::max({ EyePts.ptsLeye[0].y, EyePts.ptsLeye[1].y, EyePts.ptsLeye[2].y, EyePts.ptsLeye[3].y, EyePts.ptsLeye[4].y, EyePts.ptsLeye[5].y });

    const int MinEyeXReye = std::min({ EyePts.ptsReye[0].x, EyePts.ptsReye[1].x, EyePts.ptsReye[2].x, EyePts.ptsReye[3].x, EyePts.ptsReye[4].x, EyePts.ptsReye[5].x });
    const int MaxEyeXReye = std::max({ EyePts.ptsReye[0].x, EyePts.ptsReye[1].x, EyePts.ptsReye[2].x, EyePts.ptsReye[3].x, EyePts.ptsReye[4].x, EyePts.ptsReye[5].x });
    const int MinEyeYReye = std::min({ EyePts.ptsReye[0].y, EyePts.ptsReye[1].y, EyePts.ptsReye[2].y, EyePts.ptsReye[3].y, EyePts.ptsReye[4].y, EyePts.ptsReye[5].y });
    const int MaxEyeYReye = std::max({ EyePts.ptsReye[0].y, EyePts.ptsReye[1].y, EyePts.ptsReye[2].y, EyePts.ptsReye[3].y, EyePts.ptsReye[4].y, EyePts.ptsReye[5].y });

    cv::Range LeftEyeRangeX(MinEyeXLeye, MaxEyeXLeye);
    cv::Range LeftEyeRangeY(MinEyeYLeye, MaxEyeYLeye);

    cv::Range RightEyeRangeX(MinEyeXReye, MaxEyeXReye);
    cv::Range RightEyeRangeY(MinEyeYReye, MaxEyeYReye);

    EyeSegs.Leye = FullFrameEye(LeftEyeRangeY, LeftEyeRangeX);
    EyeSegs.Reye = FullFrameEye(RightEyeRangeY, RightEyeRangeX);

    cv::threshold(EyeSegs.Leye, EyeSegs.LeyeThres, ThresholdVal, 255, 0);
    cv::resize(EyeSegs.Leye, EyeSegs.Leye, cv::Size(), 8, 8);
    cv::resize(EyeSegs.LeyeThres, EyeSegs.LeyeThres, cv::Size(), 8, 8);

    cv::threshold(EyeSegs.Reye, EyeSegs.ReyeThres, ThresholdVal, 255, 0);
    cv::resize(EyeSegs.Reye, EyeSegs.Reye, cv::Size(), 8, 8);
    cv::resize(EyeSegs.ReyeThres, EyeSegs.ReyeThres, cv::Size(), 8, 8);

    EyeSegs.LSLeyeThres = EyeSegs.LeyeThres(cv::Range(0, EyeSegs.LeyeThres.rows), cv::Range(0, (EyeSegs.LeyeThres.cols / 2)));
    EyeSegs.RSLeyeThres = EyeSegs.LeyeThres(cv::Range(0, EyeSegs.LeyeThres.rows), cv::Range((EyeSegs.LeyeThres.cols / 2), EyeSegs.LeyeThres.cols));

    EyeSegs.LSReyeThres = EyeSegs.ReyeThres(cv::Range(0, EyeSegs.ReyeThres.rows), cv::Range(0, (EyeSegs.ReyeThres.cols / 2)));
    EyeSegs.RSReyeThres = EyeSegs.ReyeThres(cv::Range(0, EyeSegs.ReyeThres.rows), cv::Range((EyeSegs.ReyeThres.cols / 2), EyeSegs.ReyeThres.cols));

    return EyeSegs;
}

double safeDrive::calculateAvgGazeRatio(safeDrive::EyeFrameSegments& EyeSegs)
{
    int LSLeyeNumOfWhitePixels = cv::countNonZero(EyeSegs.LSLeyeThres);
    int RSLeyeNumOfWhitePixels = cv::countNonZero(EyeSegs.RSLeyeThres);
    int LeyeGazeRatio = double(LSLeyeNumOfWhitePixels) / double(RSLeyeNumOfWhitePixels);

    int LSReyeNumOfWhitePixels = cv::countNonZero(EyeSegs.LSReyeThres);
    int RSReyeNumOfWhitePixels = cv::countNonZero(EyeSegs.RSReyeThres);
    int ReyeGazeRatio = double(LSReyeNumOfWhitePixels) / double(RSReyeNumOfWhitePixels);

    double AvgGazeRatio = (LeyeGazeRatio + ReyeGazeRatio) / 2;
    return AvgGazeRatio;
}

void safeDrive::analyzeStream()
{
    cv::CascadeClassifier faceClassifier = loadFaceClassifier();
    dlib::shape_predictor FaceLandmarkDetector = loadFaceLandmarkDetector();
    std::vector<cv::Rect> faceBoxes;

    while (cap.read(Frame)) 
    {
        cv::Mat RedCalib(270, 320, CV_8UC3, cv::Scalar(0, 0, 255));
        cv::Mat GreenCalib(270, 320, CV_8UC3, cv::Scalar(0, 255, 0));
        if (CalibrationFrameCnt >= numCalibrationFrames) {
            CalibrationDone = true;
            cv::putText(RedCalib, "SafeDrive", cv::Point(70, 100), 1, 2, cv::Scalar(255, 255, 255), 2);
            cv::putText(GreenCalib, "SafeDrive", cv::Point(70, 100), 1, 2, cv::Scalar(255, 255, 255), 2);
        }
        else {
            CalibrationFrameCnt++;
            cv::putText(RedCalib, "CALIBRATION SETUP", cv::Point(50, 100), 1, 1, cv::Scalar(255, 255, 255), 2);
            cv::putText(RedCalib, std::to_string(CalibrationFrameCnt), cv::Point(135, 200), 1, 1, cv::Scalar(255, 255, 255), 2);

            cv::putText(GreenCalib, "CALIBRATION SETUP", cv::Point(50, 100), 1, 1, cv::Scalar(255, 255, 255), 2);
            cv::putText(GreenCalib, std::to_string(CalibrationFrameCnt), cv::Point(135, 200), 1, 1, cv::Scalar(255, 255, 255), 2);
        }

        //Change image to greyscale to save some computational power
        cv::cvtColor(Frame, FrameGrey, cv::COLOR_BGR2GRAY);
        faceBoxes = detectFaces(faceClassifier, FrameGrey);
        for (int i = 0; i < faceBoxes.size(); i++) {
            cv::Point P1(faceBoxes[i].x, faceBoxes[i].y);
            cv::Point P2(faceBoxes[i].x + faceBoxes[i].width, faceBoxes[i].y + faceBoxes[i].height);
            cv::rectangle(Frame, P1, P2, cv::Scalar(0, 255, 0));
            
            //wrapper so frame can be used by dlib
            dlib::cv_image<unsigned char> dlibFrameGrey(FrameGrey);
            dlib::rectangle dlibFaceBox = opencv_rect_to_dlib(faceBoxes[i]);
            dlib::full_object_detection FaceLandmarks = FaceLandmarkDetector(dlibFrameGrey, dlibFaceBox);
            EyeLandmarkPts EyePts = extractEyeLandmarks(FaceLandmarks);

            cv::polylines(Frame, &EyePts.ptsLeye, &EyePts.nptsLeye, 1, true, cv::Scalar(0, 255, 0));
            cv::polylines(Frame, &EyePts.ptsReye, &EyePts.nptsReye, 1, true, cv::Scalar(0, 255, 0));
            EyeFrameSegments EyeSegs = extractEyeSegmentsFromFrame(EyePts, FrameGrey);
            double AvgGazeRatio = calculateAvgGazeRatio(EyeSegs);

            if (AvgGazeRatio <= 0.7) {
                cv::putText(Frame, "Looking Right", cv::Point(10, 60), 1, 1, cv::Scalar(0, 0, 255), 2);
                cv::destroyWindow("Focused");
                cv::imshow("Not Focused", RedCalib);
                cv::moveWindow("Not Focused", 800, 10);
                ElevenFrameAnalysis[frameNum] = 0;
            }
            else if (AvgGazeRatio < 2.3 && AvgGazeRatio > 0.7) {
                cv::putText(Frame, "Looking central and focused", cv::Point(10, 60), 1, 1, cv::Scalar(0, 0, 255), 2);
                cv::destroyWindow("Not Focused");
                cv::imshow("Focused", GreenCalib);
                cv::moveWindow("Focused", 800, 10);
                ElevenFrameAnalysis[frameNum] = 1;

            }
            else if (AvgGazeRatio >= 2.3) {
                cv::putText(Frame, "Looking Left", cv::Point(10, 60), 1, 1, cv::Scalar(0, 0, 255), 2);
                cv::destroyWindow("Focused");
                cv::imshow("Not Focused", RedCalib);
                cv::moveWindow("Not Focused", 800, 10);
                ElevenFrameAnalysis[frameNum] = 0;

            }
            else {
                cv::putText(Frame, "Calculation Error", cv::Point(10, 60), 1, 1, cv::Scalar(0, 0, 255), 2);
                cv::destroyWindow("Focused");
                cv::imshow("Not Focused", RedCalib);
                cv::moveWindow("Not Focused", 800, 10);
                ElevenFrameAnalysis[frameNum] = 0;
            }
            cv::putText(Frame, std::to_string(AvgGazeRatio), cv::Point(10, 30), 1, 1, cv::Scalar(0, 0, 255), 2);

            cv::imshow("LEye cropped", EyeSegs.Leye);
            cv::imshow("REye cropped", EyeSegs.Reye);
            cv::imshow("LEye Threshold", EyeSegs.LeyeThres);
            cv::imshow("REye Threshold", EyeSegs.ReyeThres);

            cv::moveWindow("LEye Threshold", 150, 10);
            cv::moveWindow("REye Threshold", 500, 10);
            cv::moveWindow("LEye cropped", 150, 150);
            cv::moveWindow("REye cropped", 500, 150);
        }

        if (faceBoxes.size() == 0) {
            ElevenFrameAnalysis[frameNum] = 0;
            cv::destroyWindow("Focused");
            cv::imshow("Not Focused", RedCalib);
            cv::moveWindow("Not Focused", 800, 10);
        }
        faceBoxes.clear();

        if (frameNum == 10) {
            frameNum = 0;
            //std::cout << "Send signal to hardware" << std::endl;
            //place holder for hardware integration
        }

        cv::imshow("webcam", Frame);
        cv::moveWindow("webcam", 150, 300);

        frameNum++;
        if (cv::waitKey(1000 / fps) != -1) {
            break;
        }
    }
}
