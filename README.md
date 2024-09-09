# Blitzball Pitch Analysis Tool

## Overview
This project is a specialized video analysis tool designed for tracking and analyzing blitzball pitches. It uses computer vision techniques to detect and track the blitzball in slow-motion video footage, laying the groundwork for pitch classification based on trajectory and speed.

## Features
- Automated blitzball detection using color and motion analysis
- Frame extraction from iPhone slow-motion videos
- Interactive HSV color range calibration for optimal ball detection
- Ball trajectory tracking with bounding circle visualization

## Project Structure
The project consists of several Python scripts, each handling a specific aspect of the pitch analysis:

1. `ball_detect_copy.py`: Detects the blitzball in video frames using color and motion analysis.
2. `frame_extraction.py`: Extracts frames from iPhone slow-motion video files at specified intervals.
3. `hsv_range.py`: Provides an interactive tool for calibrating HSV color ranges for blitzball detection.

## Detailed Script Descriptions

### ball_detect_copy.py
- Implements blitzball detection using a combination of color filtering and motion detection
- Provides an interactive HSV range adjustment feature for fine-tuning ball detection
- Outputs a video with the detected blitzball highlighted by a bounding circle
- Lays the groundwork for trajectory tracking and speed calculation

### frame_extraction.py
- Extracts frames from iPhone slow-motion video files at specified intervals
- Useful for creating datasets for further analysis or machine learning tasks

### hsv_range.py
- Interactive tool for calibrating HSV color ranges
- Helps in finding the optimal color range for detecting the blitzball in various lighting conditions

## Technologies Used
- Python
- OpenCV for image processing and video handling
- NumPy for numerical operations

## Installation and Usage
[Provide instructions on how to set up the project, including required libraries and how to run each script]

## Future Improvements
- Implement full trajectory tracking of the blitzball
- Develop algorithms for pitch classification based on ball trajectory and speed
- Integrate pitch speed calculation
- Create a user interface for easier analysis and result visualization
- Expand analysis to include pitcher's form and release point

## Contributing
[Guidelines for contributing to the project]

## License
[Specify the license under which you're releasing this project]

## Contact
[Your professional contact information]

## Acknowledgements
- OpenCV community
- [Any other resources or individuals you'd like to acknowledge]