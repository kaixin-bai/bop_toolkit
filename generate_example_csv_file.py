import csv
# 新建ycbv_result.csv
with open('ycbv_result.csv', 'wb') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(["scene_id","im_id","obj_id","score","R","t","time"])
