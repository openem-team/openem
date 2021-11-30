import argparse
import json
import pandas as pd

"""
Convert openem CSV format to MS-COCO
"""

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--info", type=str, help="Path to info object")
  parser.add_argument("--supercategory", type=str, default='vehicle', help='Apply this super category')
  parser.add_argument('input_csv', type=str, help='path to input openem csv file')
  parser.add_argument('input_classes', type=str, help='path to input classes')
  parser.add_argument('output_json', type=str, help='path to output coco json')
  args = parser.parse_args()

  list_classes = pd.read_csv(args.input_classes, 
                             header=None, 
                             names=['cat_name', 'id'])
  coco_cat_idx = 1
  coco_categories = []
  coco_cat_lookup = {}
  for _,cat in list_classes.iterrows():
    coco_categories.append({"supercategory": args.supercategory,
                            'id': coco_cat_idx,
                            'name': cat.cat_name})
    coco_cat_lookup[cat.cat_name] = coco_cat_idx
    coco_cat_idx += 1
  print(coco_cat_lookup)

  annotations = pd.read_csv(args.input_csv, 
                            header=None,
                            names=['path','x1','y1', 'x2', 'y2','cat_name'])

  coco_images=[]
  coco_img_lookup = {}
  for idx,data in enumerate(annotations.path.unique()):
    coco_idx = idx+1
    coco_images.append({'file_name': data, 
                        'id': coco_idx})
    coco_img_lookup[data] = coco_idx
  
  coco_annotations = []
  for idx,data in annotations.iterrows():
    cat_id = coco_cat_lookup[data.cat_name]
    width = data.x2-data.x1
    height = data.y2-data.y1
    img_id = coco_img_lookup[data.path]
    annotation = {'image_id': img_id,
                  'bbox': [data.x1,data.y1,width,height],
                  'category_id': cat_id,
                  'id': idx+1}
    coco_annotations.append(annotation)
  coco_dataset={"categories": coco_categories,
                "images": coco_images,
                 "annotations": coco_annotations}

  if args.info:
    with open(args.input) as fp:
      coco_dataset["info"] = json.load(fp)
  with open(args.output_json,'w') as fp:
    json.dump(coco_dataset, fp, indent=4)
    



  



  


if __name__=="__main__":
  main()