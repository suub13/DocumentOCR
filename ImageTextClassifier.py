# import cv2
# import pdf2image
# import copy
# import math
#
# import numpy as np
# import torch
# import torchvision
# from torchvision.transforms import transforms
# from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
#
# import os
# import TextInImages
# import time
#
# from pdf2image import convert_from_path
#
#
# class DocLayoutParser:
#     CATEGORIES2LABELS = {
#         0: "bg",
#         1: "text",
#         2: "title",
#         3: "list",
#         4: "table",
#         5: "figure"
#     }
#     checkpoint_path = os.path.join(os.getcwd(), "model_196000.pth")
#
#     def __init__(self, model_path=None):
#         if model_path:
#             self.checkpoint_path = model_path
#
#         # 모델 가중치 파일 확인
#         if not os.path.exists(self.checkpoint_path):
#             raise Exception("Model weights not found.")
#
#         assert os.path.exists(self.checkpoint_path)
#         checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
#
#         # 클래스 수 초기화
#         self.num_classes = len(self.CATEGORIES2LABELS.keys())
#
#         # 모델 초기화 및 가중치 로드
#         self.pub_model = self._get_instance_segmentation_model(self.num_classes)
#         self.pub_model.load_state_dict(checkpoint['model'])
#         self.pub_model = self.pub_model.to("cpu")
#         self.pub_model.eval()
#
#         # 이미지 변환 정의
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.ToTensor()
#         ])
#
#     def load_and_convert_pdf(self, pdf_path):
#         pil_ori_imgs = pdf2image.convert_from_path(pdf_path)
#         self.original_images =[]
#         self.imgs_cvt = []
#         self.total_pages = len(pil_ori_imgs)
#         for pil_ori_img in pil_ori_imgs:
#             cv2_ori_img = cv2.cvtColor(np.array(pil_ori_img), cv2.COLOR_RGB2BGR)
#             self.original_images.append(cv2_ori_img)
#             self.imgs_cvt.append(cv2.cvtColor(cv2_ori_img, cv2.COLOR_BGR2RGB))
#         self.doc_shape = self.original_images[0].shape
#
#     def _get_instance_segmentation_model(self, num_classes):
#         model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
#         in_features = model.roi_heads.box_predictor.cls_score.in_features
#         model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#         in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#         hidden_layer = 256
#
#         model.roi_heads.mask_predictor = MaskRCNNPredictor(
#             in_features_mask,
#             hidden_layer,
#             num_classes
#         )
#         return model
#
#     def _preprocess_images(self):
#         self.rat = 1200 / self.imgs_cvt[0].shape[0]
#
#         prepro_imgs = []
#         for img_cvt in self.imgs_cvt:
#             prepro_img = cv2.resize(img_cvt.copy(), None, fx=self.rat, fy=self.rat)
#             prepro_img = self.transform(prepro_img).cpu()
#             prepro_imgs.append(prepro_img)
#
#         return prepro_imgs
#
#     def mult_predict_layout(self):
#         preprocessed_images = self._preprocess_images()
#         original_image_width = self.original_images[0].shape[1]
#         self.layout_info_json_list = []
#         self.layout = []
#         for preprocessed_image in preprocessed_images:
#
#             # 모델을 사용하여 예측
#             with torch.no_grad():
#                 prediction = self.pub_model([preprocessed_image])
#
#             # 예측 결과 필터링
#             mask = prediction[0]["scores"] >= 0.4
#             filtered_boxes = prediction[0]['boxes'][mask].tolist()
#             filtered_labels = list(map(lambda y: self.CATEGORIES2LABELS[y.item()], prediction[0]["labels"][mask]))
#
#             # 결과를 JSON 형태로 변환
#             layout_info_json = []
#             for label, box in zip(filtered_labels, filtered_boxes):
#                 x = math.ceil(box[0] / self.rat)
#                 y = math.ceil(box[1] / self.rat)
#                 width = math.ceil((box[2] - box[0]) / self.rat)
#                 height = math.ceil((box[3] - box[1]) / self.rat)
#
#                 layout_info_json.append({
#                     "label": label,
#                     "x": x,
#                     "y": y,
#                     "width": width,
#                     "height": height
#                 })
#             self.layout_info_json_list.append(layout_info_json)
#
#             # check layout
#             max_width = max([entry['width'] for entry in layout_info_json])
#             if max_width > original_image_width / 2:
#                 self.layout.append(1)
#             else:
#                 self.layout.append(2)
#
#
#         return self.layout_info_json_list
#
#     def expand_rectangles(self, percentage=4):
#         copy_layout_info_json_list = copy.deepcopy(self.layout_info_json_list)
#         rate = 1 + (percentage / 100)
#         self.layout_info_expand_json_list = []
#         self.non_text_json_list = []
#
#         for copy_layout_info_json in copy_layout_info_json_list:
#             temp = []
#             ntemp = []
#             for item in copy_layout_info_json:
#                 center_x = item['x'] + item['width'] / 2
#                 center_y = item['y'] + item['height'] / 2
#
#                 item['width'] *= rate
#                 item['height'] *= rate
#
#                 item['x'] = center_x - item['width'] / 2
#                 item['y'] = center_y - item['height'] / 2
#
#                 item['x'] = math.ceil(item['x'])
#                 item['y'] = math.ceil(item['y'])
#                 item['width'] = math.ceil(item['width'])
#                 item['height'] = math.ceil(item['height'])
#                 temp.append(item)
#
#                 if item['label'] in ['table', 'figure']:
#                     ntemp.append(item)
#
#             self.layout_info_expand_json_list.append(temp)
#             self.non_text_json_list.append(ntemp)
#
#         return self.layout_info_expand_json_list
#
#     def make_text_only_images(self):
#         ori_img = self.original_images[0]
#         txt_only_img = np.ones(ori_img.shape, dtype=ori_img.dtype) * 255
#         self.text_only_images = []
#
#         for idx, expand_json_list in enumerate(self.layout_info_expand_json_list):
#             tmp_img = txt_only_img.copy()
#             for expand_json in expand_json_list:
#                 if expand_json['label'] not in ['bg', 'table', 'figure']:
#                     x = expand_json['x']
#                     y = expand_json['y']
#                     width = expand_json['width']
#                     height = expand_json['height']
#                     tmp_img[y:y + height, x:x + width] = self.original_images[idx][y:y + height, x:x + width]
#             self.text_only_images.append(tmp_img)
#
#         return self.text_only_images
#
#
#
#     @staticmethod
#     def draw_bounding_box(image, json_list):
#         img = image.copy()
#         for data in json_list:
#             x_1 = int(data["x"])
#             y_1 = int(data["y"])
#             x_2 = x_1 + int(data["width"])
#             y_2 = y_1 + int(data["height"])
#
#             cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             label = data["label"]
#             cv2.putText(img, label, (x_1, y_1 - 10), font, 0.5, (0, 0, 255), 1)
#         return img
#
#
#     @staticmethod
#     def show_image(image):
#         cv2.imshow('image', image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     @staticmethod
#     def crop_image(image, x, y, width, height):
#         return image[y:y + height, x:x + width]
#
#
# if __name__ == '__main__':
#     start = time.time()
#
#     print('main')
#
#     # 모델과 이미지 경로
#     model_path = os.path.join(os.getcwd(), 'model/model_196000.pth')
#     image_path = os.path.join(os.getcwd(), 'test/Graduation Certificate_EN.pdf')
#     print("main2")
#     # 초기화 및 이미지 로드 등
#     DLP_instance = DocLayoutParser(model_path)
#     DLP_instance.load_and_convert_pdf(image_path)
#
#     print("main3")
#     # 레이아웃 예측
#     DLP_instance.mult_predict_layout()
#
#     print("main4")
#     # 레이아웃 바운딩 박스 조정 (104%)
#     DLP_instance.expand_rectangles()
#
#     print("main5")
#     # 그림들을 제외한 이미지 만들기
#     DLP_instance.make_text_only_images()
#
#     print("main6")
#     for page_num in range(DLP_instance.total_pages):
#         # 빨간 선 생성
#         red_line = np.full((DLP_instance.doc_shape[0], 5, 3), (0, 0, 0), dtype=np.uint8)
#
#         # original_image, 바운딩박스를 그린 image, text_only_image를 빨간 선으로 구분하여 합치기
#         hconcat_img = cv2.hconcat([
#             DLP_instance.original_images[page_num],
#             red_line,
#             DLP_instance.draw_bounding_box(DLP_instance.original_images[page_num],
#                                            DLP_instance.layout_info_expand_json_list[page_num]),
#             red_line,
#             DLP_instance.text_only_images[page_num]
#
#         ])
#
#         # 라사이즈 1/2를 하여 표시
#         DLP_instance.show_image(cv2.resize(hconcat_img, (int(hconcat_img.shape[1] * 0.5), int(hconcat_img.shape[0] * 0.5))))
#
#
#     # graph나 figure을 제외한 text만 있는 image
#     text_only_image_list = DLP_instance.text_only_images
# #
# # 이건 나중에 use
# #     text_only_image_list = convert_from_path('test/개인정보.pdf')
# #     text_only_image_list = text_only_image_list[10:12]
# #     print(len(text_only_image_list))
# #
# #     column 수
#     layout = DLP_instance.layout
# #     layout = []
#     # print(type(text_only_image_list[0]))
#     # for i in range(len(text_only_image_list)):
#     #     layout.append(1)
#     # print(layout)
#     # 문서 통채로 TextInImages 객체에 저장
#     extractor = TextInImages.TextInImages(text_only_image_list, layout)
#
#     # LineFormattedData를 통해 text만 추출
#     connected_text = extractor.connected_text
#     print(connected_text)
#
#     end = time.time()
#
#     print("\nIt took " + str(end - start) + " seconds")



