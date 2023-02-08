import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='ImgProcess')
parser.add_argument("--image", type=str, default="img.jpg")
parser.add_argument("--edge_min", type=int, default=30)
parser.add_argument("--edge_max", type=int, default=150)
parser.add_argument("--minLineLength", type=int, default=10)
parser.add_argument("--maxLineGap", type=int, default=10)
parser.add_argument("--line_th", type=int, default=10)
args = parser.parse_args()


def extract_edge(img, edge_min=60, edge_max=150):
   img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   img_edges = cv2.Canny(img_gray, edge_min, edge_max)
   return img_edges

def extract_line(img, line_th=10, minLineLength=100, maxLineGap=10):
   # print(f"img shape: {img.shape}")
   black_img = np.zeros(img.shape, np.uint8)
   # print(f"black img shape: {black_img.shape}")

   lines = cv2.HoughLinesP(img,1,np.pi/180,line_th,minLineLength,maxLineGap)

   for line in lines:
      x1,y1,x2,y2 = line[0]
      cv2.line(black_img,(x1,y1),(x2,y2),(255,255,255),2)
   return black_img


def main():
   img = cv2.imread(args.image)

   img_edge = extract_edge(img, edge_min=args.edge_min, edge_max=args.edge_max)
   cv2.imwrite(f'img_edgeMin{args.edge_min}_edgeMax{args.edge_max}.jpg',img_edge)
   img_line = extract_line(img_edge, line_th=args.line_th, minLineLength=args.minLineLength, maxLineGap=args.maxLineGap)

   cv2.imwrite(f'img_line_th{args.line_th}_lineMin{args.minLineLength}_MaxGap{args.maxLineGap}.jpg',img_line)


if __name__=="__main__":
   main()