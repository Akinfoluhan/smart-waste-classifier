# from cvzone.ClassificationModule import Classifier
# import cv2
#
# cap = cv2.VideoCapture(0)  # Initialize video capture
# classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
#
# while True:
#     _, img = cap.read()  # Capture frame-by-frame
#     prediction = classifier.getPrediction(img)
#     print(prediction)  # Print prediction result
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)  # Wait for a key press

import os
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2
import numpy as np
import openai

# API key
openai.api_key = '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'

cap = cv2.VideoCapture(0)
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)
classIDBin = 0

# Import all the waste images
imgWasteList = []
pathFolderWaste = "Resources/Waste"
pathList = sorted(os.listdir(pathFolderWaste), key=lambda x: int(x.split('.')[0]))
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

# Import all the bin images
imgBinsList = []
pathFolderBins = "Resources/Bins"
pathListBins = sorted(os.listdir(pathFolderBins), key=lambda x: int(x.split('.')[0]))
for path in pathListBins:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))
    print(f"Bin Image {path} shape: {imgBinsList[-1].shape}")

# Verify loaded images
print("Waste Images Loaded:")
for i, img in enumerate(imgWasteList):
    print(f"Waste Image Index {i}: {pathList[i]}")

print("Bin Images Loaded:")
for i, img in enumerate(imgBinsList):
    print(f"Bin Image Index {i}: {pathListBins[i]}")

# Class-to-bin mapping
classDic = {0: None,
            1: 0,
            2: 0,
            3: 3,
            4: 3,
            5: 1,
            6: 1,
            7: 2,
            8: 2}

# Function to classify description using GPT
def classify_with_gpt(description):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content":  "You are an expert in waste management."
                                               " Your job is to classify waste into one of the following categories:"
                                               "0: This is recyclable waste. Use the blue bin.",
    1: "This is hazardous waste. Handle carefully and use the red bin.",
    2: "This is food waste. Dispose of it in the green bin.",
    3: "This is residual waste. "},
                {"role": "user", "content": f"Classify this description: {description}"}
            ]
        )
        classification = response['choices'][0]['message']['content']
        return classification.strip()
    except Exception as e:
        print(f"Error during GPT classification: {e}")
        return None


frame_count = 0  # Add a counter
process_every = 5  # Process one frame out of every 5
while True:
    _, img = cap.read()

    imgResize = cv2.resize(img, (929, 556))  # Match dimensions of the black screen
    imgBackground = cv2.imread('Resources/background.png')
    # --- Step: Overlay the bin image here ---
    # if 0 <= classIDBin < len(imgBinsList):
    #     binImage = cv2.resize(imgBinsList[classIDBin], (150, 150))  # Resize for visibility
    #     imgBackground = cvzone.overlayPNG(imgBackground, binImage, (1000, 300))  # Top right area
    #     cv2.putText(imgBackground, f"Bin: {classIDBin}", (1000, 270),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Place the camera feed LAST so it doesn't cover other stuff
    # imgBackground[129:129 + 377, 125:125 + 665] = imgRounded
    # # --- Correct placement of bin overlay ---
    # if 0 <= classIDBin < len(imgBinsList):
    #     binImage = cv2.resize(imgBinsList[classIDBin], (150, 150))  # Resize for visibility
    #     imgBackground = cvzone.overlayPNG(imgBackground, binImage, (1000, 300))  # Safe, visible spot
    #     cv2.putText(imgBackground, f"Bin: {classIDBin}", (1000, 270),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    prediction = classifier.getPrediction(img)

    classID = prediction[1]
    print(f"Predicted Class ID: {classID}")

    if classID != 0:
        # Overlay the waste image to the right of the camera feed
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (1320, 190))

        # Overlay the arrow below the waste image
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (1380, 400))

        # Map to the appropriate bin class
        classIDBin = classDic.get(classID, 0)
        print(f"Mapped Bin Class: {classIDBin}")

    # Overlay the bin image if valid
    # if 0 <= classIDBin < len(imgBinsList):
    #     cv2.imshow("Bin Image", imgBinsList[classIDBin])
    #     # DEBUG: Show the bin image in a separate window
    #     # cv2.imshow("Test Bin Image", imgBinsList[classIDBin])
    #     # cv2.waitKey(0)
    #     binImage = cv2.resize(imgBinsList[classIDBin], (150, 150))  # Resize for visibility
    #     imgBackground = cvzone.overlayPNG(imgBackground, binImage, (50, 650))  # Move to bottom-left
    #
    #     # imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (1300, 500))
    #     print(f"Overlaying bin image: {classIDBin}")
    # if 0 <= classIDBin < len(imgBinsList):
    #     binImage = cv2.resize(imgBinsList[classIDBin], (150, 150))  # Resize for better visibility
    #
    #     # OPTION 1: Place under the arrow (adjust if needed)
    #     imgBackground = cvzone.overlayPNG(imgBackground, binImage, (1380, 560))
    #
    #     # OPTIONAL: Debug print
    #     print(f"Overlaying bin image: {classIDBin}")

    else:
        print(f"Error: classIDBin {classIDBin} out of bounds!")

    # Resize the camera feed
    # imgResize = cv2.resize(img, (874, 516))
    # mask = np.zeros((516, 874, 3), dtype=np.uint8)
    # radius = 50
    # color = (255, 255, 255)
    # cv2.rectangle(mask, (0, 0), (874, 516), color, thickness=-1)
    # kernel_size = max(1, (radius // 2) * 2 + 1)
    # mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    # imgRounded = cv2.bitwise_and(imgResize, mask)
    # imgBackground[220:220 + 500, 143:143 + 874] = cv2.resize(imgRounded, (874, 500))

    # Resize the camera feed to fit inside the iMac screen
    imgResize = cv2.resize(img, (665, 377))  # Width x Height

    # Optional: Apply smooth rounded corners
    mask = np.zeros((377, 665, 3), dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (665, 377), (255, 255, 255), thickness=-1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    imgRounded = cv2.bitwise_and(imgResize, mask)

    # Paste it perfectly inside the black screen area of the iMac
    imgBackground[129:129 + 377, 125:125 + 665] = imgRounded

    # Updated: Show bin only when classID is not 0 and bin index is valid
    if classID != 0 and 0 <= classIDBin < len(imgBinsList):
        binImage = cv2.resize(imgBinsList[classIDBin], (150, 150))
        imgBackground = cvzone.overlayPNG(imgBackground, binImage, (1000, 300))
        cv2.putText(imgBackground, f"Bin: {classIDBin}", (1000, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # Function to draw a prettier box with rounded corners
    def draw_pretty_box(img, x, y, w, h, color=(0, 0, 0), corner_radius=20):
        # Draw the rounded rectangle
        overlay = img.copy()
        output = img.copy()

        # Top-left corner
        cv2.ellipse(overlay, (x + corner_radius, y + corner_radius), (corner_radius, corner_radius), 180, 0, 90, color,
                    -1)
        # Top-right corner
        cv2.ellipse(overlay, (x + w - corner_radius, y + corner_radius), (corner_radius, corner_radius), 270, 0, 90,
                    color, -1)
        # Bottom-left corner
        cv2.ellipse(overlay, (x + corner_radius, y + h - corner_radius), (corner_radius, corner_radius), 90, 0, 90,
                    color, -1)
        # Bottom-right corner
        cv2.ellipse(overlay, (x + w - corner_radius, y + h - corner_radius), (corner_radius, corner_radius), 0, 0, 90,
                    color, -1)

        # Draw rectangles to fill gaps
        cv2.rectangle(overlay, (x + corner_radius, y), (x + w - corner_radius, y + h), color, -1)
        cv2.rectangle(overlay, (x, y + corner_radius), (x + w, y + h - corner_radius), color, -1)

        # Apply the overlay with transparency
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output


    # Add GPT message to the screen with a prettier box
    message_box_x, message_box_y = 50, 750
    message_box_w, message_box_h = 1250, 80
    imgBackground = draw_pretty_box(imgBackground, message_box_x, message_box_y, message_box_w, message_box_h,
                                    color=(0, 0, 0))



    # Display the output
    cv2.imshow("Output", imgBackground)

    # Quit on 'q' key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

