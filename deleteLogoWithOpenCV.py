
import numpy as np
from PIL.Image import Image
from pdf2image import convert_from_path
import cv2
from FormattedTextInImages import TextInImages
import time

print("start")
start = time.time()

input_path = 'Graduation Certificate_KR'
language = 'kor+eng'

if "KR" in input_path:
    language = 'kor+eng'
elif "EN" in input_path:
    language = 'eng+kor'

dpi = 70
threshold = 200
converted_images = []
images = convert_from_path(f'test/{input_path}/{input_path}.pdf', dpi=dpi)



#  1.294:1 letter (8.5 x 11)
# 1:1.41 A4     (8.27 x 11.69)

for img in images:
    width, height = img.size
    print(width, height)
    width = int(width * 2)
    height = int(height * 2)
    print(width, height)

    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    _, thresholded_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

    converted_images.append(thresholded_img)

    # cv2.imshow("thresholded_img", thresholded_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

for image_index in range(len(converted_images)):
    output_image_path = f"test/{input_path}/{threshold}_{image_index+1}.png"  # PNG 형식으로 저장
    cv2.imwrite(output_image_path, converted_images[image_index])

layout = []
for i in range(len(converted_images)):
    layout.append(1)

# 문서 통채로 TextInImages 객체에 저장
extractor = TextInImages(converted_images, layout, language)

# LineFormattedData를 통해 text만 추출
connected_text = extractor.connected_text
# print(connected_text)

with open(f"test/{input_path}/{threshold}_dpi={dpi}_reduced_expanded.txt", 'w') as f:
    f.write(connected_text)


end = time.time()
print(f"it took {end - start} seconds")
