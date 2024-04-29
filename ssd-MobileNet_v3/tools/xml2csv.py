import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    print(os.getcwd())
    # image_path = os.path.join(os.getcwd(), 'dataset/crack/test')  # 根据自己实际目录修改，或者使用下面的路径
    image_path = 'C:/Users/ASUS/Desktop/intern/glovebox/start/datasets/mydata/xml/train'
    print(image_path)
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('C:/Users/ASUS/Desktop/intern/glovebox/start/datasets/train.csv', index=None)  # 根据自己实际目录修改
    print('Successfully converted xml to csv.')

main()




