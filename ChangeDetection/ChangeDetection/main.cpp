//
//  main.cpp
//  ChangeDetection
//
//  Created by Chris Radford on 2018-01-08.
//  Copyright © 2018 Chris Radford. All rights reserved.
//  Built and tested in Xcode V:9.2(9C40b) OS MacOS Sierra V:10.12.6
//
//  PURPOSE: This program will detect motion/change within the field of view of the provided camera.
//  If change is detected an image will be savied to a specified valid folder inputed at lauch. Images
//  will continue to be taken if change is detected at an interval specified at launch. Program will attempting to
//  re-initalize reference frame based on refresh rate  given. Will not refresh if change detected and will attempt
//  at next refresh rate increment.
//
//  INPUTS: (all inputs read by xcode as string and will be convereted by program)
//  filePath type: String - a valid filepath ending in a /(MAC) or \(PC) depending on OS
//  refreshRate type: int - rate (in frames) of referene frame being initalized. MUST BE < 800.
//  saveRate type: int - rate (in seconds) at which images are saved if change detected.
//
//  OUTPUTS:
//  Image type: .jpg - image save d to given valid folder/filepath. Name based on date and time frame captured.
//
//  PROGRAM REQUIREMENTS AND INSTALLATION FOR PC 64 BIT MACHINE USING VISUAL STUDIO 2017:
//  OpenCV v3.1 or later with approriate linker path established.
//  All instructions referenced from https://www.youtube.com/watch?v=l4372qtZ4dc&t=372s
//  1. This PC properties advanced stystem settings (Control Panel -> System Security -> System)
//      1.a - Go to Environement variables
//      1.8 - System Vriables -> Path -> Edit
//      1.b - Link in folder in OpenCV build (...OpenCVLocal\build\X64\vc14\bin)
//  2. Visual Studio New Project
//      2.a - Visual C++ Empty Project
//      2.b - Name users choise; ***Location - USE DEFAULT PROVIDED***
//      2.C - Click Ok
//  3.  In top menu bar select Debug and x64 from respective drop down settings (might say Debug, x86)
//  4. Load the Cpp file.
//      4.a - In solution Explorer Right CLick and add existing file
//      4.b locate this file and add.
//  5. Linking Libraries
//      5.a - In Solution Explorer right clhick project -> Select Properties
//      5.b - C/C++ -> General -> Additional Include Directories
//          4.b.i - locate OpenCV include folder (...OpenCVLocal\build\include)
//      5.c - Linker -> General -> Additional Library Directories
//          4.c.i - locate OpenCv lib folder (...OpenCVLocal\build\X64\vc14\lib)
//      5.d - Linker -> Input -> Additional Dependencies
//          4.d.i - Locate opencv_worldd file and paste name (opencv_world310d.lib)
//  6. Configuration Properties -> Debugging ->Command Arguments
//      5.a - Insert input command argument variables delimited by a space
//  7. Click Apply and OK

//  SETUP:
//  1. Upon running program it will have a still frame where you will select as many ROIs (0-infinity) by pressing on a
//     location within the image that will be a corner then dragging dianonally across while mouse is pressed. To set opposite
//     corner lift mouse and click spacebar. Process can be repeated many times. Press ESC to continue
//  2. Next section is used to tune the erode and dialate functions. adjust as needed then press ESC to continue

#include <opencv2/opencv.hpp>
#include <ctime>
#include <string>
#include <tuple>
#include <list>
#include <iostream>
#include <fstream>


using namespace std;
using namespace cv;

//Global Variables
Point P1(0,0);
Point P2(0,0);
bool complete = false;

//Trackbar variables
const int dilateSliderMax = 20;
int dilateSlider;
const int erodeSliderMax = 20;
int erodeSlider;

struct MouseParams
{
    Mat img;
    Point pt;
    int lucky;
};


void onMouse ( int event, int x, int y, int d, void *ptr ){
    switch(event){
            //mouse pressed (get coordinate of first point)
        case CV_EVENT_LBUTTONDOWN:
            //cout << "press" << endl;
            P1.x=x;
            P1.y=y;
            break;
            //mouse lifted (get coordinate of final point)
        case CV_EVENT_LBUTTONUP:
            //cout << "lift" << endl;
            P2.x=x;
            P2.y=y;
            complete = true;
            break;
        default:
            break;
    }
    //If lifted mouse and have two valid points()
    if(complete && P1 != P2){
        //cout << "Building ROI" << endl;
        Rect*ROI = (Rect*)ptr;
        int w,h;
        x = min(P1.x, P2.x);
        y = min(P1.y, P2.y);
        h= abs(P1.y-P2.y);
        w = abs(P1.x-P2.x);
        
        ROI->x=x;
        ROI->y=y;
        ROI->width=w;
        ROI->height=h;
        cout << *ROI << endl;
        complete = false;
    }
}

vector<Rect> ROIsetup(Mat frame){
    ofstream outFile;
    vector<Rect> MasterROI;
    Rect ROI;
    char key;
    namedWindow("ROI");
    setMouseCallback("ROI",onMouse, &ROI);
    while(true){
        //cout << ROI << endl;
        MasterROI.push_back (ROI);
        //draw ROI grabbed in onMouse
        rectangle( frame, ROI, Scalar(0,255,0), 1, 8, 0 );
        //cout << "drawing" <<endl;
        imshow("ROI",frame);
        moveWindow("ROI", 0,50);
        key = (char)waitKey(0);   // explicit cast
        if (key == 27) break;                // break if `esc' key was pressed.
        if (key == ' ') continue;;
    }
    destroyAllWindows();
    return MasterROI;
}

tuple<Mat, Mat,vector<vector<Point>>,vector<Vec4i>> getContours(Mat firstFrame, Mat gray){
    //Trackbar variables
    Mat imageDifference, thresh;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    absdiff(firstFrame, gray, imageDifference);
    threshold(imageDifference, thresh, 25, 255, THRESH_BINARY);
    erode(thresh, thresh, Mat(),Point(-1, -1), erodeSlider,1,1);
    dilate(thresh, thresh, Mat(), Point(-1, -1), dilateSlider, 1, 1);
    findContours(thresh, contours, hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    return{ thresh, imageDifference,contours,hierarchy};
}

tuple<Mat, bool, bool> insertContours(Mat frame, int minArea, vector<vector<Point>> contours, bool occupied, bool initialFrame){
    int foundContourCount;
    Rect boundRect, intersectingRect;
    if (contours.size() > 1){
        foundContourCount = 0;
        
        
        for(int i = 0; i < contours.size(); i++){
            //get the boundboxes and save the ROI as an Image
            if (contourArea(contours[i]) < minArea){
                continue;
            }
            boundRect = boundingRect( Mat(contours[i]));
            rectangle( frame, boundRect.tl(), boundRect.br(), Scalar(0,255,0), 1, 8, 0 );
            foundContourCount++;
        }
        //if changes were significant enough
        if(foundContourCount > 1){
            occupied = true;
        }
        //if changes not significant enough
        else{
            occupied = false;
            initialFrame = true;
        }
    }
    //if no contours founds
    else{
        occupied = false;
        initialFrame = true;
        
    }
    return{frame,occupied,initialFrame};
}

tuple<Mat, bool, bool> checkContours(Mat frame, int minArea, vector<vector<Point>> contours, bool occupied, bool initialFrame,vector<Rect> masterROI){
    int foundContourCount;
    Rect boundRect, intersectingRect;
    vector<Rect> boundRectList;
    if (contours.size() > 1){
        foundContourCount = 0;
        
        //remove all contours with too small an area
        for(int i = 0; i < contours.size(); i++){
            //get the boundboxes and save the ROI as an Image
            if (contourArea(contours[i]) < minArea){
                continue;
            }
            boundRect = boundingRect( Mat(contours[i]));
            boundRectList.push_back(boundRect);
        }
        
        //check if valid bournding areas intersect the provided ROI's
        for(int i = 0; i < masterROI.size();i++){
            rectangle( frame, masterROI[i].tl(), masterROI[i].br(), Scalar(255,255,0), 4, 8, 0 );
            cout << masterROI[i] << endl;
            for(int ii = 0; ii < boundRectList.size();ii++){
                intersectingRect = boundRectList[ii] & masterROI[i];
                if(intersectingRect.area() > 0){
                    rectangle( frame, boundRectList[ii].tl(), boundRectList[ii].br(), Scalar(0,255,0), 1, 8, 0 );
                    foundContourCount++;
                }
            }
        }
        
        //if changes significant and fell within stated area
        if(foundContourCount > 1){
            occupied = true;
        }
        //if changes not significant enough and/or didnt fall within stated area
        else{
            occupied = false;
            initialFrame = true;
        }
    }
    //if no contours founds
    else{
        occupied = false;
        initialFrame = true;
        
    }
    return{frame,occupied,initialFrame};
}

Mat tuneDisplay(Mat frame, Mat thresh, Mat imageDifference){
    Rect roi;
    
    
    resize(frame, frame, Size(frame.cols*0.8,frame.rows*0.8),INTER_AREA);
    Mat master(Size(frame.cols,frame.rows*1.5),frame.type(),Scalar::all(0));
    //resize and re-format
    resize(imageDifference, imageDifference, Size(frame.cols/2,frame.rows/2),INTER_AREA);
    resize(thresh, thresh, Size(frame.cols/2,frame.rows/2),INTER_AREA);
    cvtColor(imageDifference, imageDifference, CV_GRAY2RGB);
    cvtColor(thresh, thresh, CV_GRAY2RGB);
    
    //image difference (TL)
    roi = Rect (0,0,imageDifference.cols,imageDifference.rows);
    imageDifference.copyTo(master(roi));
    //image difference (TR)
    roi = Rect (thresh.cols,0,thresh.cols,thresh.rows);
    thresh.copyTo(master(roi));
    //image difference (B)
    roi = Rect (0,imageDifference.rows,frame.cols,frame.rows);
    frame.copyTo(master(roi));
    
    return master;
}

int main(int argc, const char * argv[]) {
    //declared variables
    vector<Vec4i> hierarchy;
    vector<Rect> MasterROI;
    Mat firstFrame, frame, gray,thresh,imageDifference, master;
    int height, cnt = 0, minArea = 500;
    bool occupied = false, initialFrame = true;
    vector<vector<Point>> contours;
    string filePath = argv[1];
    int refreshRate = stoi(argv[2]);
    int saveRate = stoi(argv[3]);
    string text, saveLocal;
    char timeString[100], timeFile[100];
    struct tm *timeptr;
    time_t currtime,initialtimeStep;
    Mat frameToSave;
    Rect fullFrame;
    //---------------------------------------------------------
    //-------------------------ROI Selection-------------------------------------
    //---------------------------------------------------------
    //get camera operational and make sure working correctly
    VideoCapture camera(0);
    if(!camera.isOpened()){
        cout << "cannot open camera" << endl;
        return(1);
    }
    //prevents grabbing the first frame
    while(cnt < 10){
        camera.read(frame);
        cnt++;
    }
    //Begin ROI selection
    camera.read(frame);
    MasterROI = ROIsetup(frame);
    
    //erase first dud ROI in masterROI
    MasterROI.erase(MasterROI.begin());
    
    cout << MasterROI.size() << endl;
    //if ROI is empty create one of frame.
    if (MasterROI.size() < 1){
        fullFrame = Rect (0,0, frame.cols,frame.rows);
        MasterROI.push_back(fullFrame);
        
    }
    
    // Create Trackbars
    namedWindow("Debugger",1);
    char TrackbarName[50];
    sprintf( TrackbarName, "Dilate x %d", dilateSliderMax );
    createTrackbar( TrackbarName, "Debugger", &dilateSlider, dilateSliderMax);
    sprintf( TrackbarName, "Erode x %d", erodeSliderMax );
    createTrackbar( TrackbarName, "Debugger", &erodeSlider, erodeSliderMax);
    //---------------------------------------------------------
    //------------------------Menu selection-------------------------------------
    //---------------------------------------------------------
    while(true){
        //keep refreshing frame count
        if (cnt == 4000)
            cnt = 10;
        
        //grab frame from camera
        camera.read(frame);
        height = frame.rows;
        if(frame.empty()){
            cout << "frame was not captured correctly. Aborting" << endl;
            return(2);
        }
        
        //pre processing
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size( 21, 21 ), 0, 0 );
        
        //initrialize first frame if necessary
        if(firstFrame.empty()){
            if(cnt < 10){
                continue;
            }
            cout << "grabbed initial reference frame " << endl;
            gray.copyTo(firstFrame);
            continue;
        }
        
        tie(thresh, imageDifference,contours,hierarchy) = getContours(firstFrame, gray);
        tie(frame,occupied,initialFrame) = insertContours(frame, minArea, contours, occupied, initialFrame);
        master = tuneDisplay(frame, thresh, imageDifference);
        
        //Display currFrame
        imshow("Debugger", master);
        //moveWindow("Debugger", 0,50);
        if (waitKey(30) >= 0)
            break;
        
    }
    destroyAllWindows();
    //---------------------------------------------------------
    //------------------------Main Program-------------------------------------
    //---------------------------------------------------------
    while(true){
        //variable incrementation and reset
        currtime = time(NULL);
        timeptr = localtime(&currtime);
        cnt++;
        
        //keep refreshing frame count
        if (cnt == 4000)
            cnt = 10;
        
        //grab frame from camera
        camera.read(frame);
        height = frame.rows;
        frame.copyTo(frameToSave);
        if(frame.empty()){
            cout << "frame was not captured correctly. Aborting" << endl;
            return(2);
        }
        
        //pre processing
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size( 21, 21 ), 0, 0 );
        
        //initrialize first frame if necessary
        if(firstFrame.empty()){
            if(cnt < 10){
                continue;
            }
            cout << "grabbed initial reference frame " << endl;
            gray.copyTo(firstFrame);
            continue;
        }
        
        //generate and filter contours
        tie(thresh, imageDifference,contours,hierarchy) = getContours(firstFrame, gray);
        tie(frame,occupied,initialFrame) = checkContours(frame, minArea, contours, occupied, initialFrame, MasterROI);
        
        //if change detetected and need to save frame
        if(occupied){
            //first instance of a frame
            if(initialFrame == true){
                initialFrame = false;
                initialtimeStep = currtime;
                strftime(timeFile, sizeof(timeFile),"%F-%H-%M-%S", timeptr);
                saveLocal = filePath+timeFile+".jpg";
                imwrite(saveLocal, frameToSave);
                cout << "grabbed initial frame" << endl;
            }
            else if ((initialtimeStep + saveRate) == currtime ){
                initialtimeStep = currtime;
                strftime(timeFile, sizeof(timeFile),"%F-%H-%M-%S", timeptr);
                saveLocal = filePath+timeFile+".jpg";
                imwrite(saveLocal, frameToSave);
                cout << "grabbed frame based on step" << endl;
            }
        }
        //update reference frame provided image isn't occupied or force reset
        if (cnt % refreshRate ==0 && !occupied){
            cout << "Standard frame refresh" << endl;
            gray.copyTo(firstFrame);
            continue;
        }
        
        //draw everything to output frames
        if(occupied){
            text = "Occupied";
        }
        else{
            text = "Unoccupied";
            
        }
        putText(frame, text, Point2f(10,50), FONT_HERSHEY_SIMPLEX, 2,  Scalar(0,255,255,255),8);
        strftime(timeString, sizeof(timeString),"%A %d %b %Y %T ", timeptr);
        putText(frame, timeString, Point2f(10,height-10), FONT_HERSHEY_SIMPLEX, 1,  Scalar(0,0,255,255),2);
        
        imshow("Security feed", frame);
        moveWindow("Security feed", 20,20);
        if (waitKey(30) >= 0)
            break;
    }
    camera.release();
    destroyAllWindows();
    return(0);
}


