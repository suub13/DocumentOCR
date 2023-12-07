import string
import pytesseract
import os

# Replace '/path/to/tessdata' with the actual path to your tessdata directory
os.environ['TESSDATA_PREFIX'] = os.path.join(os.getcwd(), "venv/lib/python3.10/site-packages/tessdata/")


class TextInImages:

    def __init__(self, text_only_image_list, layout, lang):
        self.text_only_image_list = text_only_image_list
        # self.non_text_list = non_text_list
        self.layout = layout
        self.lang = lang

        # 이거 create_text_list call한 값으로 check_connecting_line 호출하면 될듯
        self.extracted_data_list = self.create_data_list(self.text_only_image_list, self.layout)
        self.connected_text = self.connect_text(self.extracted_data_list)

    def create_data_list(self, text_only_image_list, layout):
        extracted_data_list = []

        # 페이지 별로 text 추출 및 text 와 non_text의 위치 정보를 합침
        for idx in range(len(text_only_image_list)):
            text_in_image = LineFormattedData(text_only_image_list[idx], layout[idx], self.lang)
            extracted_data_list += text_in_image.line_formatted_data

        return extracted_data_list

    def connect_text(self, extracted_data_list):
        connected_text = ''
        # for idx, txt in enumerate(extracted_data_list[:-1]):
        #     if not((txt.rstrip()[-1] in string.punctuation) | (extracted_data_list[idx+1][0].isupper())):
        #         txt = txt.rstrip() + ' '
        #
        #     connected_text += txt
        #
        # connected_text += extracted_data_list[-1]

        # return connected_text

        for i in extracted_data_list:
            connected_text += i
        return connected_text



class LineFormattedData:
    def __init__(self, page, layout, lang='eng'):
        self.page = page
        self.height = page.shape[0]
        self.width = page.shape[1]
        # self.width = page.size[0]
        # self.height = page.size[1]
        self.layout = layout
        self.line_formatted_data = self.process_pdf(page, lang)

    def process_pdf(self, page, lang):
        # layout 수에 따라, 개행 적용된 text 추출

        extracted_data = self.extract_data_from_page(page, lang, 1)
        line_formatted = ''

        # if self.layout == 1:
        #     line_formatted = self.data_to_text_json_1(extracted_data)
        # elif self.layout == 2:
        #     line_formatted = self.data_to_text_json_2(extracted_data)
        # else:
        #     print("Invalid column_num")

        # return line_formatted
        return extracted_data

    def extract_data_from_page(self, page, language, confidence=-1):
        # empty 데이터와 정확성 낮은 데이터 filtering 후 필요한 데이터 만 추출

        # os.environ['TESSDATA_PREFIX'] = 'tessdata'



        # data = pytesseract.image_to_data(page, lang='kor', output_type=pytesseract.Output.DICT)
        data = pytesseract.image_to_string(page, lang=language)

        # extracted_data = []
        # for i in range(len(data['text'])):
        #     text = data['text'][i].strip()
        #     if (data['conf'][i] > confidence) & (text != ''):
        #         extracted_data.append({
        #             "text": text,
        #             "x": data['left'][i],
        #             "y": data['top'][i],
        #             "width": data['width'][i],
        #             "height": data['height'][i],
        #             "line_num": data['line_num'][i],
        #             "block_num": data['block_num'][i],
        #             "par_num": data['par_num'][i],
        #         })

        # return extracted_data
        return data


    def calculate_spacing_height(self, extracted_data):
        # spacing 계산
        max_height = max([word['height'] for word in extracted_data])

        total_gap = 0.0
        count = 0
        avg_gap = 1.25 * max_height

        for i, data in enumerate(extracted_data[:-1]):  # block_num 이 같지 않으면, 개행 됐을 수 있음
            if (data['block_num'] == extracted_data[i + 1]['block_num']) & \
                    (data['par_num'] == extracted_data[i + 1]['par_num']) & \
                    (data['line_num'] != extracted_data[i + 1]['line_num']):
                total_gap += extracted_data[i + 1]['y'] - data['y']
                count += 1

        if count != 0:
            avg_gap = round(total_gap / count, 2)

        return avg_gap, max_height

    def data_to_text_json_1(self, words):
        # layout 1 에서 text 만 추출 - 개행 적용
        self.spacing, max_height = self.calculate_spacing_height(words)

        formatted_text = ""

        self.left_x = min([word['x'] for word in words])
        self.right_x = max([(word['x'] + word['width']) for word in words])

        self.top_y = min([word['y'] for word in words])
        self.bottom_y = max([word['y'] for word in words]) + self.spacing

        # 한 줄씩 나누기 위한 parameters
        total_line_formatted = []

        for idx, word in enumerate(words[:-1]):
            if (word['text'] != '-') & (word['text'][-1] == '-'):
                word['text'] = word['text'][:-1]
            next_word = words[idx + 1]
            formatted_text += word['text'] + ' '

            # 블락이 다를 때 또는 같은데 paragraph 가 다를 경우 개행
            if (word['block_num'] != next_word['block_num']) \
                    | ((word['block_num'] == next_word['block_num']) & (word['par_num'] != next_word['par_num'])):
                pass

            # 블락이 같고 paragraph 도 같은데, 라인이 다르면 뒤에 다음 단어 width 이상 남으면 개행
            elif (word['block_num'] == next_word['block_num']) & (word['par_num'] == next_word['par_num']) \
                    & (word['line_num'] != next_word['line_num']) & \
                    (self.right_x - word['x'] - word['width'] > next_word['width']):
                pass

            # 라인이 다른데 다음 줄이 알파벳 으로 시작 하지 않으면 개행
            elif (word['line_num'] != next_word['line_num']) & (not next_word['text'][0].isalpha()):
                pass
            else:
                continue

            if not((formatted_text.startswith("Fig")) | (formatted_text.startswith("Table"))):
                total_line_formatted.append(formatted_text + '\n')
            formatted_text = ''

        formatted_text += words[-1]['text'] + '\n'
        total_line_formatted.append(formatted_text)

        return total_line_formatted

    def data_to_text_json_2(self, words):
        # 2 column layout 에서 text 만 추출 - 개행 적용
        self.spacing, max_height = self.calculate_spacing_height(words)

        total_line_formatted = []
        formatted_text = ""

        middle_line_x = (self.width) / 2

        self.first_column_left_x = min([word['x'] for word in words])
        self.first_column_right_x = max(
            [(word['x'] + word['width']) for word in words if (word['x'] + word['width']) < middle_line_x])

        right_words_filtered = [word['x'] for word in words if (word['x'] + word['width']) > middle_line_x]
        if not right_words_filtered:
            self.second_column_left_x = middle_line_x
        else:
            self.second_column_left_x = min(right_words_filtered)

        self.second_column_right_x = max([(word['x'] + word['width']) for word in words])


        self.top_y = min([word['y'] for word in words])
        self.bottom_y = max([word['y'] for word in words]) + self.spacing

        for idx, word in enumerate(words[:-1]):
            if (word['text'] != '-') & (word['text'][-1] == '-'):
                word['text'] = word['text'][:-1]
            if word['text'] == 'e':
                word['text'] = '-'
            next_word = words[idx + 1]
            formatted_text += word['text'] + ' '

            if word['x'] < middle_line_x:
                criteria = self.first_column_right_x
            else:
                criteria = self.second_column_right_x

            # 블락이 다를 때 또는 같은데 paragraph 가 다를 경우 개행
            if (word['block_num'] != next_word['block_num']) \
                    | ((word['block_num'] == next_word['block_num']) & (word['par_num'] != next_word['par_num'])):
                pass

            # 블락이 같고 paragraph 도 같은데, 라인이 다르면 뒤에 다음 단어 이상 남으면 개행
            elif (word['block_num'] == next_word['block_num']) & (word['par_num'] == next_word['par_num']) \
                    & (word['line_num'] != next_word['line_num']) \
                    & (criteria - (word['x'] + word['width']) > next_word['width']):
                pass

            # 라인이 다른데 다음 줄이 알파벳 으로 시작 하지 않으면 개행
            elif (word['line_num'] != next_word['line_num']) & (not next_word['text'][0].isalpha()) \
                    & ((word['par_num'] != next_word['par_num']) | (word['block_num'] != next_word['block_num'])):
                pass
            else:
                continue

            if not((formatted_text.startswith("Fig")) | (formatted_text.startswith("Table"))):
                total_line_formatted.append(formatted_text + '\n')
            formatted_text = ''

        # 마지막 단어 추가
        formatted_text += words[-1]['text']
        total_line_formatted.append(formatted_text + '\n')

        return total_line_formatted

