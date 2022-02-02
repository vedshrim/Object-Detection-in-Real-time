import cv2
from gui_buttons import Buttons

button = Buttons()
button.add_button("person", 20, 20) 
button.add_button("cell phone", 20, 100)
button.add_button("bottle", 20, 180)
colors = button.colors


net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(380,380), scale=1/255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("objects list")
print(classes)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



def click_button(event, x, y, flags, param):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)
        

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)
while True:
    ret, frame = cap.read()
    
    active_buttons = button.active_buttons_list()
    print("Active Buttons", active_buttons)
    
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x , y, width, height) = bbox
        class_name = classes[class_id]
        color = colors[class_id]
        
        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
    
    #print("class ids", class_ids)
    #print("scores", scores)
    #print("bboxes", bboxes)
    
    button.display_buttons(frame)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()