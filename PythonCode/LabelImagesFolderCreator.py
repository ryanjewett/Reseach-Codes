import cv2
from ultralytics import YOLO
import os
import random


class ImageLabelingFolder():
    def __init__(self, 
                 modelname="yolov8x.pt", 
                 savefolder="unlabeled1", 
                 videofolder="RAW_VIDEO",
                 totalimages=100,
                 randomizefolder=True,
                 inferanceimprovement=False, 
                 idealclasses=['car', 'truck', 'bicycle', 'person', 'motorcycle', 'bus'],
                 confthreashold=0.5, 
                 density=1.0,
                 fpsscale=1) -> None:
        
        self.model = modelname
        self.inferanceimprovement = inferanceimprovement

        self.savefolder = savefolder
        self.videofolder = videofolder
        self.videonamelist = []
        self.totalimages = totalimages
        self.randomizefolder = randomizefolder
        self.tempfolder = "TEMP_FOLDER"

        self.confthreashold = confthreashold
        self.density = density
        self.fpsscale = fpsscale
        self.idealclasses = idealclasses

    def videoSetup(self):  # gets all valid videos from videofolder
        
        if not os.path.exists(self.videofolder):
            print(f"Folder {self.videofolder} does not exist.")
            return
        
        for videoname in os.listdir(self.videofolder):
            if videoname.endswith(('.mp4', '.avi', '.mov')):
                self.videonamelist.append(os.path.join(self.videofolder, videoname))

        print(f"Found {len(self.videonamelist)} valid video(s): {self.videonamelist}")

    def seperateVideo(self, singlevideoname):
        if not os.path.exists(self.tempfolder):
            os.makedirs(self.tempfolder)
        
        video_capture = cv2.VideoCapture(singlevideoname)
        original_fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / self.fpsscale)
        frame_count = 0
        saved_count = 0
        
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_filename = f"{self.tempfolder}/frame_{saved_count}.jpg"
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
            
            frame_count += 1
        
        video_capture.release()
        cv2.destroyAllWindows()
        print(f"Extracted {saved_count} frames to the folder: {self.tempfolder}")

    def clearTempFolder(self):
        try:
            for item in os.listdir(self.tempfolder):
                item_path = os.path.join(self.tempfolder, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            print(f"Folder {self.tempfolder} Cleared")
        except Exception as e:
            print(f"Error occurred: {e}")

    def folderInferance(self):
        model = YOLO(self.model)
        image_scores = {}
        
        for image in os.listdir(self.tempfolder):
            imagepath = os.path.join(self.tempfolder, image)
            imageresults = []
            results = model.predict(imagepath)
            results = results[0]

            for box in results.boxes:
                class_id = str(results.names[box.cls[0].item()])
                conf = round(box.conf[0].item(), 2)
                if class_id in self.idealclasses and conf >= self.confthreashold:
                    imageresults.append((class_id, conf))

            if imageresults:
                score = self.calculateScore(imageresults)
                image_scores[imagepath] = score
        
        self.imageEvaluation(image_scores)

    def calculateScore(self, imageresults):
        score = 0
        for class_id, conf in imageresults:
            if class_id in self.idealclasses:
                score += conf  # You can also weigh the scores differently if needed
        return score

    def imageEvaluation(self, image_scores):
        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        selected_images = sorted_images[:self.totalimages]

        if not os.path.exists(self.savefolder):
            os.makedirs(self.savefolder)

        for i, (imagepath, score) in enumerate(selected_images):
            image_save_path = os.path.join(self.savefolder, f"selected_frame_{i}.jpg")
            cv2.imwrite(image_save_path, cv2.imread(imagepath))
            print(f"Saved {image_save_path} with score {score}")

        if self.randomizefolder:
            self.randomizeOutputFolder()

    def randomizeOutputFolder(self):
        images = os.listdir(self.savefolder)
        random.shuffle(images)

        for i, image in enumerate(images):
            src = os.path.join(self.savefolder, image)
            dst = os.path.join(self.savefolder, f"randomized_frame_{i}.jpg")
            os.rename(src, dst)

        print(f"Randomized images in the folder {self.savefolder}")

    def run(self):
        self.videoSetup()
        for videoname in self.videonamelist:
            self.seperateVideo(videoname)
            self.folderInferance()
            self.clearTempFolder()

# Example of running the class
if __name__ == "__main__":
    img_labeler = ImageLabelingFolder()
    img_labeler.run()
