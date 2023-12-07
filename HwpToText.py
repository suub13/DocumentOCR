
# 1번째 방식
#
import PyPDF2

pdfReader = PyPDF2.PdfFileReader("test/test.pdf")
print(" No. Of Pages :", pdfReader.numPages)

total = ''
for i in range(pdfReader.numPages):
    data = pdfReader.getPage(i).extractText()
    total += data

print(total)


# 2번째 방식

# # pip install olefile
# import olefile
# f = olefile.OleFileIO('test/content.hwp')
# encoded_text = f.openstream('PrvText').read()
# decoded_text = encoded_text.decode('utf-16')
#
# print(len(encoded_text))
# print(len(decoded_text))
# with open("result.txt", 'w') as f:
#     f.write(decoded_text)
#
# print(decoded_text)


# 3 번째 방식
#
# # terminal에서 진행
# # pip install pyhwp
# # pip install six
# # hwp5txt --output "저장되는파일.txt" "hwp위치.hwp"
#
import subprocess

# Install the required packages using pip
subprocess.run(["pip", "install", "pyhwp"])
subprocess.run(["pip", "install", "six"])

# Run the hwp5txt command
hwp_location = "test/content.hwp"  # Replace with the actual path to your HWP file
output_file = "test1.txt"  # Replace with the desired output file name

subprocess.run(["hwp5txt", "--output", output_file, hwp_location])

print(f"Conversion complete. Text saved as {output_file}")