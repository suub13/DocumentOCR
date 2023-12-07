import os
import time

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import cv2
import TextInImages
import ImageTextClassiferForColoring


def is_black_or_gray(color, rgb):
    """주어진 색상이 검정색 또는 회색조인지 확인합니다."""
    r, g, b, _ = color  # RGBA에서 마지막 값(알파)은 무시합니다
    return 0 <= r <= rgb and 0 <= g <= rgb and 0 <= b <= rgb

def process_page(page, rgb, zoom):
    """주어진 페이지를 고해상도 이미지로 변환하고 처리합니다."""
    # 페이지를 더 높은 해상도로 렌더링
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 이미지 처리
    img = img.convert("RGBA")
    datas = img.getdata()

    new_data = []
    for item in datas:
        # 검정색과 회색조를 제외한 모든 색상을 하얗게 변경
        if is_black_or_gray(item, rgb):
            new_data.append(item)
        else:
            new_data.append((255, 255, 255, 255))

    img.putdata(new_data)
    # RGBA에서 RGB로 변환
    img = img.convert("RGB")
    img = img.resize((int(img.width / (zoom/2)), int(img.height / (zoom/2))), Image.ANTIALIAS)

    return img


def process_pdf(input_pdf, rgb, zoom):
    """PDF 파일을 읽고, 각 페이지에 대해 이미지 처리를 수행한 후, 결과를 저장합니다."""
    doc = fitz.open(input_pdf)
    output_images = []

    for page in doc:
        img = process_page(page, rgb, zoom)
        output_images.append(img)

    layout = []

    for i in range(len(output_images)):
        layout.append(1)
    print(layout)
    # 문서 통채로 TextInImages 객체에 저장
    extractor = TextInImages.TextInImages(output_images, layout)

    # LineFormattedData를 통해 text만 추출
    connected_text = extractor.connected_text
    print(connected_text)

    return output_images


if __name__ == '__main__':

    start = time.time()

    zoom = 10
    rgb = 210

    input_pdf = "test/Graduation Certificate_KR.pdf"
    output_pdf = "test/original_{zoom}.pdf"

    output_images = process_pdf(input_pdf, rgb, zoom)
    if output_images:
        output_images[0].save(output_pdf, save_all=True, append_images=output_images[1:], format="PDF")
    for image_index in range(len(output_images)):
        output_image_path = f"test/color_{rgb}_{image_index+1}.png"  # PNG 형식으로 저장
        output_images[image_index].save(output_image_path)

    end = time.time()
    print("\nIt took " + str(end - start) + " seconds")

    # image를 읽었을 떄 깨지는 지 확인하기 위한 코드

    # doc = fitz.open(input_pdf)
    # output = []
    # for page in doc:
    #     mat = fitz.Matrix(zoom, zoom)
    #     pix = page.get_pixmap(matrix=mat)
    #     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    #     img = img.convert("RGBA")
    #     img = img.convert("RGB")
    #     output_image_path = f"test/original_{zoom}.png"  # PNG 형식으로 저장
    #     img.save(output_image_path)
    #     output.append(img)
    # output[0].save(output_pdf, save_all=True, append_images=output[1:], format="PDF")


    #     start = time.time()
    #
    #     print('main')
    #
    # #     # 모델과 이미지 경로
    #     model_path = os.path.join(os.getcwd(), 'model/model_196000.pth')
    #     image_path = os.path.join(os.getcwd(), 'test/Graduation Certificate_EN.pdf')
    #     print("main2")
    #
    #     clean_images = process_pdf(image_path)
    #
    # #     # 초기화 및 이미지 로드 등
    #     DLP_instance = ImageTextClassiferForColoring.DocLayoutParser(model_path)
    #     DLP_instance.load_and_convert_pdf(image_path)
    # #
    # #     print("main3")
    # #     # 레이아웃 예측
    # #     DLP_instance.mult_predict_layout()
    # #
    # #     print("main4")
    # #     # 레이아웃 바운딩 박스 조정 (104%)
    #     DLP_instance.expand_rectangles()
    # #
    # #     print("main5")
    # #     # 그림들을 제외한 이미지 만들기
    #     DLP_instance.make_text_only_images()
    # #
    # #     print("main6")
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
    # #
    # #         # 라사이즈 1/2를 하여 표시
    #         DLP_instance.show_image(cv2.resize(hconcat_img, (int(hconcat_img.shape[1] * 0.5), int(hconcat_img.shape[0] * 0.5))))
    # #
    # #
    # #     # graph나 figure을 제외한 text만 있는 image
    #     text_only_image_list = DLP_instance.text_only_images
    # # #
    # # # 이건 나중에 use
    # # #     text_only_image_list = convert_from_path('test/개인정보.pdf')
    # # #     text_only_image_list = text_only_image_list[10:12]
    # # #     print(len(text_only_image_list))
    # # #
    # #     column 수
    #     layout = DLP_instance.layout
    # #     layout = []
    #     # print(type(text_only_image_list[0]))
    #     # for i in range(len(text_only_image_list)):
    #     #     layout.append(1)
    #     # print(layout)
    # #     # 문서 통채로 TextInImages 객체에 저장
    #     extractor = TextInImages.TextInImages(text_only_image_list, layout)
    # #
    # #     # LineFormattedData를 통해 text만 추출
    #     connected_text = extractor.connected_text
    #     print(connected_text)
    #
    #     end = time.time()
    #
    #     print("\nIt took " + str(end - start) + " seconds")

