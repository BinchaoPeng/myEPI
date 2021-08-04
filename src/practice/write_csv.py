import csv
import os

current_dir = os.path.abspath('.')
file_name = os.path.join(current_dir, "csss.csv")
csvfile = open(file_name, 'wt', newline='')  # encoding='utf-8'

writer = csv.writer(csvfile, delimiter=",")
header = ['uel', 'title']
csvrow1 = []
csvrow2 = []
csvrow1.append("测试1")
csvrow1.append("测试2")
csvrow2.append("111")
csvrow2.append("222")

writer.writerow(header)
writer.writerows(zip(csvrow1, csvrow2))

csvfile.close()

def main():
    current_dir = os.path.abspath('.')
    file_name = os.path.join(current_dir, "csss.csv")
    csvfile = open(file_name, 'wt', newline='')  # encoding='utf-8'

    writer = csv.writer(csvfile, delimiter=",")
    header = ['uel', 'title']
    csvrow1 = []
    csvrow2 = []
    csvrow1.append("测试1")
    csvrow1.append("测试2")
    csvrow2.append("111")
    csvrow2.append("222")

    writer.writerow(header)
    writer.writerows(zip(csvrow1, csvrow2))

    csvfile.close()


if __name__ == '__main__':
    main()
