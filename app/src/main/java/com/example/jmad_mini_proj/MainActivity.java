package com.example.jmad_mini_proj;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private ImageView imageView;
    private Button detectButton;

    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error if OpenCV is not loaded
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        detectButton = findViewById(R.id.detectButton);

        detectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                detectLicensePlate();
            }
        });
    }

    private void detectLicensePlate() {
        // Load the image from resources or file path
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.img1);

        // Convert the bitmap to Mat (OpenCV) format
        Mat imageMat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC4);
        Utils.bitmapToMat(bitmap, imageMat);

        // Convert the image to grayscale
        Mat gray = new Mat();
        Imgproc.cvtColor(imageMat, gray, Imgproc.COLOR_RGBA2GRAY);

        // Apply Gaussian blur to reduce noise
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

        // Apply edge detection using the Canny algorithm
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, 50, 150);

        // Find contours in the edge-detected image
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Filter contours based on area to identify potential character regions
        int minArea = 300;  // Minimum area for a contour to be considered as a character
        int maxArea = 1000;  // Maximum area for a contour to be considered as a character

        List<MatOfPoint> characterContours = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > minArea && area < maxArea) {
                characterContours.add(contour);
            }
        }

// Sort character contours from left to right based on their x-coordinate
        Collections.sort(characterContours, (contour1, contour2) -> {
            Rect rect1 = Imgproc.boundingRect(contour1);
            Rect rect2 = Imgproc.boundingRect(contour2);
            return Integer.compare(rect1.x, rect2.x);
        });

// Group close contours as part of the same number plate contour
        List<MatOfPoint> numberPlateContours = new ArrayList<>();
        int distanceThreshold = 50;  // Threshold for grouping contours

        for (int i = 0; i < characterContours.size(); i++) {
            MatOfPoint contour = characterContours.get(i);
            Rect rect = Imgproc.boundingRect(contour);

            // Check the distance between the current contour and the previous one
            if (i > 0) {
                Rect prevRect = Imgproc.boundingRect(characterContours.get(i - 1));
                int distance = rect.x - (prevRect.x + prevRect.width);

                if (distance <= distanceThreshold) {
                    // Contours are close, consider them as part of the same number plate
                    numberPlateContours.remove(numberPlateContours.size() - 1);  // Remove the previous contour from number plate contours
                    rect.x = prevRect.x;  // Update the x-coordinate of the current contour to include the previous contour
                }
            }

            numberPlateContours.add(new MatOfPoint(new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height), new Point(rect.x, rect.y + rect.height)));
        }

// Draw the number plate contours on the image
        for (MatOfPoint contour : numberPlateContours) {
            Imgproc.drawContours(imageMat, Collections.singletonList(contour), 0, new Scalar(0, 255, 0), 2);
        }

        // Draw bounding boxes around license plate candidates
        Scalar color = new Scalar(0, 255, 0);
        int thickness = 2;

        for (MatOfPoint contour : numberPlateContours) {
            // Get the bounding rectangle of the contour
            MatOfPoint2f contour2f = new MatOfPoint2f();
            contour.convertTo(contour2f, CvType.CV_32FC2);
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approxCurve, 0.02 * Imgproc.arcLength(contour2f, true), true);
            MatOfPoint approx = new MatOfPoint();
            approxCurve.convertTo(approx, CvType.CV_32S);

            org.opencv.core.Rect rect = Imgproc.boundingRect(approx);

            // Draw the bounding rectangle on the image
            Imgproc.rectangle(imageMat, rect.tl(), rect.br(), color, thickness);
        }

        // Convert the processed Mat back to Bitmap for display
        Bitmap resultBitmap = Bitmap.createBitmap(imageMat.cols(), imageMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imageMat, resultBitmap);

        // Display the result image on the ImageView
        imageView.setImageBitmap(resultBitmap);
    }
}