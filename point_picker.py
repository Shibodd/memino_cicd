import numpy as np
import cv2 as cv
import time
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("vid")
argparser.add_argument("output")
argparser.add_argument("-s", "--scale", default=1, type=float)
args = argparser.parse_args()

class Recorder:
  def __init__(self):
    self.frames = {}
    self.points = []
    self.frame = None

  def add_point(self, pos):
    assert(self.frame is not None)

    self.points.append(pos)

    if len(self.points) >= 4:
      self.frames[self.frame] = self.points
      self.points = []
      self.frame = None
      return True
    
    return False

  def start_frame(self, frame):
    self.points = []
    self.frame = frame

cap = cv.VideoCapture(args.vid)
rec = Recorder()

def next_frame():
  global cap, args

  ret, frame = cap.read()
  if not ret:
    cap.release()
    return False
  
  rec.start_frame(cap.get(cv.CAP_PROP_POS_FRAMES))
  
  frame = cv.resize(frame, None, fx=args.scale, fy=args.scale)
  cv.imshow('frame', frame)

def click_event(event, x, y, *_):
  if event == cv.EVENT_LBUTTONDOWN:
    if rec.add_point(np.array([x / args.scale, y / args.scale])):
      next_frame()

cv.namedWindow('frame')
cv.setMouseCallback('frame', click_event)

next_frame()

while cap.isOpened():
  key = cv.waitKey(1)

  if key == ord('q'):
    cap.release()
    break

  if key == ord(' '):
    next_frame()

cap.release()
cv.destroyAllWindows()

def format_point(pt):
  return f"{pt[0]},{pt[1]}"

with open(args.output, "wt") as f:
  f.write("id,x0,y0,x1,y1,x2,y2,x3,y3\n")
  f.writelines(
    f"{frame_id},{pts[0][0]},{pts[0][1]},{pts[1][0]},{pts[1][1]},{pts[2][0]},{pts[2][1]},{pts[3][0]},{pts[3][1]}\n"
    for frame_id, pts in rec.frames.items()
  )