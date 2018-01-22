#include"iostream"
#include "Main.h"
#include "pthread.h"
#include <cstdlib>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2)
{
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	//cout << "i " << i << " j " << j << endl;
	return (i < j);
}

void *ReadImage(void *threadid) {
	long tid;
	tid = (long)threadid;


	Mat imgOriginalScene = imread("NP.jpg");


//	imgOriginalScene = cv::imread("NP4.jpg");

	if (imgOriginalScene.empty()) {
		std::cout << "error: image not read from file\n\n";
		_getch();
		return(0);
	}

	std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);

	vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);

	//cv::imshow("imgOriginalScene", imgOriginalScene);         

	if (vectorOfPossiblePlates.empty()) {
		std::cout << std::endl << "no license plates were detected" << std::endl;
		system("pause");
	}
	else {



		std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);

		PossiblePlate licPlate = vectorOfPossiblePlates.front();

		if (licPlate.strChars.length() == 0) {
			std::cout << std::endl << "no characters were detected" << std::endl << std::endl;
			return(0);
		}

		//drawRedRectangleAroundPlate(imgOriginalScene, licPlate);                

		std::cout << std::endl << "license plate read from image = " << licPlate.strChars << std::endl;
		std::cout << std::endl << "-----------------------------------------" << std::endl;

		// writeLicensePlateCharsOnImage(imgOriginalScene, licPlate);             

		//cv::imshow("imgOriginalScene", imgOriginalScene);

		cv::imwrite("imgOriginalScene.png", imgOriginalScene);
	}

	cv::waitKey(0);


	pthread_exit(NULL);
	return 0;
}
void *CameraStream(void *threadid) {
	long tid;
	tid = (long)threadid;
	pthread_t thread;
	
	

	int r = 0, c = 0;
	Mat imgOriginalScene;
	VideoCapture capture(1);
	int q;

	while (cvWaitKey(30) != 'q')
	{
	capture >> imgOriginalScene;
	if (true)
	{



	//Mat imgOriginalScene = imread("NP.jpg");
	Mat frame;
	//namedWindow("image", WINDOW_NORMAL);
	//imshow("image 0", img);
	Mat gray;
	cvtColor(imgOriginalScene, gray, CV_BGR2GRAY);
	//namedWindow("Grayimage", WINDOW_NORMAL);
	//imshow("Grayimage 1", gray);
	Mat clear;
	bilateralFilter(gray, clear, 9, 75, 75);
	//namedWindow("Remove Noise", WINDOW_NORMAL);
	//imshow("Remove Noise 3", clear);
	Mat hist;
	equalizeHist(clear, hist);
	//namedWindow("Histogram", WINDOW_NORMAL);
	//imshow("Histogram 4", hist);
	Mat morph;


	for (int i = 1; i < 10; i++)
	{

	morphologyEx(hist, morph, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(5, 5)), Point(-1, -1), i);
	}
	//namedWindow("Morphology 5", WINDOW_NORMAL);
	//imshow("Morphology 5", morph);


	Mat morph_image = hist - morph;
	//namedWindow("Subtracted", WINDOW_NORMAL);
	//imshow("Subtracted 6", morph_image);
	Mat thresh;
	threshold(morph_image, thresh, 0, 255, THRESH_OTSU);
	//imshow("thresold 7", thresh);

	Mat edge;
	Canny(thresh, edge, 250, 255, 3);
	//imshow("canny edge 8", edge);

	Mat dil;
	dilate(edge, dil, getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));
	//imshow("dilation 9", dil);


	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;


	findContours(dil, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	sort(contours.begin(), contours.end(), compareContourAreas);

	vector<Point> approx;
	vector<Point> screenCnt;

	for (size_t i = 0; i < contours.size(); i++) {
	approxPolyDP((Mat(contours[i])), approx, arcLength(Mat(contours[i]), true) * 0.06, true);
	//cout << "size = " << approx.size() << "    " << arcLength(Mat(contours[i]), true)* 0.06 << endl;
	double a = contourArea(contours[i], false);


	if (a < 6505.5 && a>5000 && approx.size() == 4) {
	screenCnt = approx;
	cout << a << endl;

	drawContours(imgOriginalScene, std::vector<std::vector<cv::Point>>{screenCnt}, 0, Scalar(0, 255, 255), 3);
	cout << "size arc =    " << arcLength(Mat(contours[i]), true) << endl;

	Mat mask = Mat::zeros(imgOriginalScene.rows, imgOriginalScene.cols, CV_8UC1);
	r = imgOriginalScene.rows;
	c = imgOriginalScene.cols;

	Rect re = boundingRect(contours[i]);

	cout << "x " << re.x << "y " << re.x << " hight =    " << re.height << "width =    " << re.width << endl;



	drawContours(mask, std::vector<std::vector<cv::Point>>{screenCnt}, -1, Scalar(255), CV_FILLED);

	Mat crop(imgOriginalScene.rows, imgOriginalScene.cols, CV_8UC3);

	crop.setTo(Scalar(255, 255, 255));

	imgOriginalScene.copyTo(crop, mask);

	normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC3);
	//imshow("mask", mask);
	//imshow("cropped", crop);

	Rect cr = Rect(re.x, re.y, re.width, re.height);

	Rect cr1 = Rect(re.x, re.y / 2, re.width, re.height / 2);
	cout << " crop x " << cr1.x << " crop y  " << cr1.y << " crop  hight =    " << cr1.height << "width =    " << cr1.width << endl;
	Mat cuted = crop(cr1);
	Mat cut = crop(cr);

	Size size(100, 100);//the dst image size,e.g.100x100
	Mat dst;//dst image
	Mat src;//src image

	//imshow("Display cut window", dst);
	cv::RotatedRect box = cv::minAreaRect(cv::Mat(screenCnt));
	cv::Mat rot_mat = cv::getRotationMatrix2D(box.center, 45, 1);


	cvtColor(cut, cut, CV_BGR2GRAY);
	imshow("gray number plate window", cut);
	imwrite("NP.jpg", cut);
	int rc = pthread_create(&thread, NULL, ReadImage, (void *)1);


	if (rc) {
		cout << "Error:unable to create thread," << rc << endl;
		exit(-1);
	}


	/*Mat tc = imread("NP.png");
	Rect m = Rect(0, 50, 75, 75);
	Mat ct = cut(m);


	imshow("half gray number plate window", ct);
	*/

	break;

	}
	}


	imshow("Display window", imgOriginalScene);


	}
	}

	
	pthread_exit(NULL);
	return 0;
}




int main(void) {

    bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN(); 
    if (blnKNNTrainingSuccessful == false) {                            
        std::cout << std::endl << std::endl << "error: error: KNN traning was not successful" << std::endl << std::endl;
        return(0);                                                    
    }

   // cv::Mat imgOriginalScene;          
	pthread_t threads;
	int rc = pthread_create(&threads, NULL, CameraStream, (void *)0);
	if (rc) {
		cout << "Error:unable to create thread," << rc << endl;
		exit(-1);
	}
	system("pause");
    return(0);
}


void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
    cv::Point2f p2fRectPoints[4];

    licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);            

    for (int i = 0; i < 4; i++) {                                      
        cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_RED, 2);
    }
}


void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
    cv::Point ptCenterOfTextArea;                   
    cv::Point ptLowerLeftTextOrigin;               

    int intFontFace = CV_FONT_HERSHEY_SIMPLEX;                              
    double dblFontScale = (double)licPlate.imgPlate.rows / 30.0;            
    int intFontThickness = (int)round(dblFontScale * 1.5);            
    int intBaseline = 0;

    cv::Size textSize = cv::getTextSize(licPlate.strChars, intFontFace, dblFontScale, intFontThickness, &intBaseline);    

    ptCenterOfTextArea.x = (int)licPlate.rrLocationOfPlateInScene.center.x;         

    if (licPlate.rrLocationOfPlateInScene.center.y < (imgOriginalScene.rows * 0.75)) {      
                                                                                           
        ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) + (int)std::round((double)licPlate.imgPlate.rows * 1.6);
    }
    else {                                                                                
        ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) - (int)std::round((double)licPlate.imgPlate.rows * 1.6);
    }

    ptLowerLeftTextOrigin.x = (int)(ptCenterOfTextArea.x - (textSize.width / 2));           
    ptLowerLeftTextOrigin.y = (int)(ptCenterOfTextArea.y + (textSize.height / 2));          

                                                                                          
    cv::putText(imgOriginalScene, licPlate.strChars, ptLowerLeftTextOrigin, intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
}


