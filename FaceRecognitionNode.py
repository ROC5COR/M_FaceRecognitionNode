import threading
import falcon
from wsgiref import simple_server
import os
import json
import pickle
import cv2
import numpy as np

NODE_IDENTITY = "FaceRecognitionNode"

class FaceRecognition:
    def __init__(self, embedder_model, recognizer_file, label_encoder_file):
        print("[INFO] loading face recognizer...")
        self.embedder = cv2.dnn.readNetFromTorch(embedder_model)

        self.recognizer = pickle.loads(open(recognizer_file, "rb").read())
        self.le = pickle.loads(open(label_encoder_file, "rb").read())

    def process_from_jpg(self, jpg_data):
        img_data = cv2.imdecode(jpg_data, cv2.IMREAD_UNCHANGED)
        return self.process(img_data)

    def process(self, face_data):
        """
        In this function we supppose img_data is already a numpy array created by cv2
        """
        (fH, fW) = face_data.shape[:2]

        # ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
            print("Image is too small")
            return None

        faceBlob = cv2.dnn.blobFromImage(face_data, 1.0 / 255,
            (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.embedder.setInput(faceBlob)
        vec = self.embedder.forward()

        # perform classification to recognize the face
        preds = self.recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        if proba < 0.7:
            return None
        
        pred = self.le.classes_[j]
        return pred



class FaceRecognitionNode(threading.Thread):
    def __init__(self, port):
        threading.Thread.__init__(self)
        
        current_file_dirname = os.path.dirname(os.path.abspath(__file__))
        embedder_model = os.path.join(current_file_dirname, 'openface_nn4.small2.v1.t7') # Can be loaded from Git
        
        recognizer_file = os.environ.get('FaceRecognitionNode_Recognizer', None) # Path to the custom recognition pickle file
        label_encoder_file = os.environ.get('FaceRecognitionNode_Label', None) # Path to the custom label encoder pickle file

        if recognizer_file is None:
            print("Missing custom recognizer pickle file")
            return

        if label_encoder_file is None:
            print("Missing custom label encoder pickle file")
            return

        face_recognition = FaceRecognition(embedder_model, recognizer_file, label_encoder_file)

        class MainRoute:
            def on_get(self, req ,res):
                res.content_type = 'plain/text'
                res.body = "FaceRecognitionNode<br>/output_text"

        class Process:
            def on_get(self, req, res):
                res.content_type = 'plain/text'
                res.body = "You have to POST an image 'data' as data, use 'from' field to specify 'rgb' or 'jpg'"
            def on_post(self, req , res):
                img_data = req.bounded_stream.read()
                params = req.params

                if img_data is not None and img_data is not b'':
                    if 'from' in params:
                        if params['from'] == 'jpg':
                            face_recognition.process_from_jpg(img_data)
                            res.content_type = 'application/json'
                            res.body = json.dumps({'status':'ok'})
                        elif params['from'] == 'rgb':
                            rgb_img = np.array(json.loads(img_data), np.uint8)
                            results = face_recognition.process(rgb_img)
                            res.content_type = 'application/json'
                            res.body = json.dumps({'status':'ok', 'data':results})
                        else:
                            print("Don't know how to process 'from': ",params['from'])
                            res.content_type = 'application/json'
                            res.body = json.dumps({'status':'ko', 'error':'from parameter not recognized either rgb or jpg'})
                    else:
                        res.content_type = 'application/json'
                        res.body = json.dumps({'status':'ko', 'error':'You have to specify a from field in param either rgb or jpg'})
                else:
                    res.content_type = 'application/json'
                    res.body = json.dumps({'status':'ko', 'error':'Missing image data'})

        
        api = falcon.API()
        api.add_route('/', MainRoute())
        api.add_route('/process', Process())
        self.server = simple_server.make_server('', port, app=api)

    def run(self):
        print("[FaceRecognitionNode:INFO] Server started")
        self.server.serve_forever()


if __name__ == '__main__':
    os.environ['FaceRecognitionNode_Recognizer'] = 'FaceRecognitionNode/custom_recognizer.pickle'
    os.environ['FaceRecognitionNode_Label'] = 'FaceRecognitionNode/custom_label_encoder.pickle'

    frn = FaceRecognitionNode(8099)

    frn.start()