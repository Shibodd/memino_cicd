import numpy as np
import cv2 as cv
import time
import argparse
import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument("vid", type=str)
argparser.add_argument("points", type=str)
argparser.add_argument("img1", type=str)
argparser.add_argument("img2", type=str)
argparser.add_argument("frame", type=int)
argparser.add_argument("output", type=str)
argparser.add_argument("-s", "--scale", default=1, type=float)
args = argparser.parse_args()

cap = cv.VideoCapture(args.vid)

fourcc = cv.VideoWriter_fourcc(*'H264')
res = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

writ = cv.VideoWriter(args.output, fourcc, cap.get(cv.CAP_PROP_FPS), res, True)


def interp(df: pd.DataFrame, frame):
  gte = df[df.index.values >= frame]
  lte = df[df.index.values <= frame]

  # Don't extrapolate
  if gte.shape[0] == 0:
    return lte.iloc[-1]
  if lte.shape[0] == 0:
    return gte.iloc[0]
  
  gte = gte.iloc[0]
  lte = lte.iloc[0]

  # If we have an exact position
  if gte.name == frame:
    return gte
  if lte.name == frame:
    return lte

  # 1D interp
  t = (frame - lte.name) / (gte.name - lte.name)
  return lte + t * (gte - lte)


def get_pts(df):
  pts = []
  for i in range(4):
    pts.append(np.array([
      df[f"x{i}"],
      df[f"y{i}"]
    ], dtype=np.float32))
  return np.array(pts, dtype=np.float32)


img1 = cv.imread(args.img1)
img2 = cv.imread(args.img2)

df = pd.read_csv("points.csv", index_col="id")
pts_src1 = np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]]], dtype=np.float32)
pts_src2 = np.array([[0, 0], [img2.shape[1], 0], [img2.shape[1], img2.shape[0]], [0, img2.shape[0]]], dtype=np.float32)

cv.namedWindow('frame')


while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    cap.release()
    break

  key = cv.waitKey(int(1000 / 15))

  if key == ord('q'):
    break

  frame_id = cap.get(cv.CAP_PROP_POS_FRAMES)
  pts_dst = get_pts(interp(df, frame_id))

  mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
  cv.fillConvexPoly(mask, pts_dst.astype(int), 255)
  mask_inv = cv.bitwise_not(mask)

  if frame_id < args.frame:
    pts_src = pts_src1
    img = img1
  else:
    pts_src = pts_src2
    img = img2

  matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
  warped = cv.warpPerspective(img, matrix, (frame.shape[1], frame.shape[0]))
  warped = cv.bitwise_and(warped, warped, mask=mask)

  frame = cv.bitwise_and(frame, frame, mask=mask_inv)

  frame = cv.add(frame, warped)
  writ.write(frame)
  cv.imshow('frame', frame)

writ.release()
cap.release()
cv.destroyAllWindows()
