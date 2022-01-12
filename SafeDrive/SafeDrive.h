///////////////////////////////////////////////////
// Written by: Abdulrahman Elgendy
// Last Update: 12-01-2022
//////////////////////////////////////////////////

#pragma once
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/opencv.h>


class safeDrive 
{
public:
    safeDrive(int deviceId);
    void analyzeStream();

private:
    void thresholdTrackBar();
    cv::CascadeClassifier loadFaceClassifier();
    std::vector<cv::Rect> detectFaces(cv::CascadeClassifier& faceClassifier, const cv::Mat& Frame);
    dlib::rectangle opencv_rect_to_dlib(cv::Rect rectangle);
    dlib::shape_predictor loadFaceLandmarkDetector();

    std::vector<cv::Point> LeftEyeContour;
    std::vector<cv::Point> RightEyeContour;

    cv::VideoCapture cap;

    struct EyeLandmarkPts
    {
        const cv::Point* ptsLeye;
        const cv::Point* ptsReye;
        int nptsLeye;
        int nptsReye;
    };

    EyeLandmarkPts extractEyeLandmarks(dlib::full_object_detection& shape);
    cv::Mat Frame;
    cv::Mat FrameGrey;
 

    struct EyeFrameSegments
    {
        cv::Mat Leye;
        cv::Mat Reye;
        cv::Mat LeyeThres;
        cv::Mat ReyeThres;
        cv::Mat LSLeyeThres;
        cv::Mat RSLeyeThres;
        cv::Mat LSReyeThres;
        cv::Mat RSReyeThres;
    };

    EyeFrameSegments extractEyeSegmentsFromFrame(EyeLandmarkPts& EyePts, cv::Mat& FrameGrey);
    double calculateAvgGazeRatio(EyeFrameSegments& EyeSegs);

    const int ThresholdMax = 255;
    int ThresholdVal = 50;

    int frameNum = 0;
    const int fps = 20;
    const int numCalibrationFrames = 1000;

    int CalibrationFrameCnt = 0;
    bool CalibrationDone = false;
    int ElevenFrameAnalysis[11] = {};
    int NumOfConsecutiveNegInstructions = 0;

    cv::Mat FrameResize;
    cv::Mat FrameGreyEqualized;

    double VerticalSegDistLeye;
    double HorizontalSegDistLeye;
    double SegRatioLeye;

    double VerticalSegDistReye;
    double HorizontalSegDistReye;
    double SegRatioReye;

};