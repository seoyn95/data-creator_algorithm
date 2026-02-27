# 파일명에서 성별 및 스타일 정보를 마지막 인덱스부터 추출하는 함수
def parse_label_from_filename(filename):
    parts = filename.split('_')
    gender = '여성' if parts[-2].startswith('W') else '남성'  
    style = parts[-3]  
    return gender, style

# 이미지 폴더 내 유효한 이미지ID 목록을 수집하는 함수
def load_image_ids(image_dir):
    image_ids = set()
    for root, _, files in os.walk(image_dir):
        for file_name in files:
            if file_name.endswith('.jpg'):
                image_id = file_name.split('_')[1]  # 이미지ID 추출
                image_ids.add(image_id)
    return image_ids

# 유효한 라벨 파일에 대한 성별 및 스타일별 통계를 수집하는 함수
def collect_label_statistics(label_dir, image_ids):
    statistics = defaultdict(lambda: defaultdict(int))
    for root, _, files in os.walk(label_dir):
        for file_name in files:
            if file_name.endswith('.json'):
                image_id = file_name.split('_')[1] 
                if image_id in image_ids: 
                    gender, style = parse_label_from_filename(file_name)
                    statistics[gender][style] += 1
    return statistics

# 통계 출력 함수
def display_statistics(statistics, dataset_name):
    print(f"\n{dataset_name} 데이터 통계:")
    print(f"{'성별':<10} {'스타일':<15} {'이미지 수':<10}")
    print("=" * 35)
    total_images = 0
    for gender, styles in statistics.items():
        for style, count in styles.items():
            print(f"{gender:<10} {style:<15} {count:<10}")
            total_images += count
    print(f"{dataset_name} 총 이미지 수: {total_images}")
    return total_images

# Training 및 Validation 데이터 경로 설정
train_image_dir = '/content/drive/MyDrive/unzipped_data_creator_camp/training_image'  
train_label_dir = '/content/drive/MyDrive/unzipped_data_creator_camp/training_label' 
val_image_dir = '/content/drive/MyDrive/unzipped_data_creator_camp/validation_image'
val_label_dir = '/content/drive/MyDrive/unzipped_data_creator_camp/validation_label'  


# Training 데이터 통계 수집
train_image_ids = load_image_ids(train_image_dir)
train_statistics = collect_label_statistics(train_label_dir, train_image_ids)
train_total = display_statistics(train_statistics, "Training")

# Validation 데이터 통계 수집
val_image_ids = load_image_ids(val_image_dir)
val_statistics = collect_label_statistics(val_label_dir, val_image_ids)
val_total = display_statistics(val_statistics, "Validation")

print(f"Validation 데이터 총 이미지 수: {val_total}")


import os
import json
import pandas as pd

# 데이터 리스트 초기화
data = []

# 경로 설정
train_image_dir = '/content/drive/MyDrive/unzipped_data_creator_camp/training_image'
train_label_dir = '/content/drive/MyDrive/unzipped_data_creator_camp/training_label'

# 이미지 및 JSON 파일 이름 추출
image_files = {f.split('_')[1]: f for f in os.listdir(train_image_dir) if f.endswith('.jpg')}
json_files = {f.split('_')[1]: f for f in os.listdir(train_label_dir) if f.endswith('.json')}

# 이미지 파일과 JSON 파일 매칭 및 데이터 추출
for key, img_file in image_files.items():
    if key in json_files:
        json_file = json_files[key]
        json_path = os.path.join(train_label_dir, json_file)

        try:
            # JSON 파일 열기 및 데이터 로드
            with open(json_path, 'r') as file:
                json_data = json.load(file)

            # 응답자 ID 및 스타일 선호 정보 추출
            user_id = json_data['user']['R_id']
            img_name = json_data['item']['imgName']
            preference = json_data['item']['survey']['Q5']

            # 선호도 정보 설정
            style_preferred = img_name if preference == 2 else ''
            style_not_preferred = img_name if preference == 1 else ''

            # 리스트에 정보 추가
            data.append({
                '응답자 ID': user_id,
                '스타일 선호': style_preferred,
                '스타일 비선호': style_not_preferred
            })

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error processing file {json_path}: {e}")
            continue

# 데이터 프레임 생성
df = pd.DataFrame(data)

# 중복된 ID에 대한 데이터 합치기
def combine_styles(series):
    return ', '.join(set(filter(None, series)))

# 응답자 ID로 그룹화하고 스타일 데이터를 합치기
df = df.groupby('응답자 ID').agg({
    '스타일 선호': combine_styles,
    '스타일 비선호': combine_styles
}).reset_index()

# 결과 출력
df



# 데이터 리스트 초기화
data2 = []

val_image_dir = '/content/drive/MyDrive/unzipped_data_creator_camp/validation_image'  # 검증 이미지 경로
val_label_dir = '/content/drive/MyDrive/unzipped_data_creator_camp/validation_label'  # 검증 라벨 경로

# JPG 파일 읽기 (이미지 파일과 JSON 파일을 매칭하기 위해)
image_files = {f.split('_')[1]: f for f in os.listdir(val_image_dir) if f.endswith('.jpg')}
json_files = {f.split('_')[1]: f for f in os.listdir(val_label_dir) if f.endswith('.json')}

# 이미지 파일과 JSON 파일 매칭 및 데이터 추출
for key, img_file in image_files.items():
    if key in json_files:
        json_file = json_files[key]
        json_path = os.path.join(train_label_dir, json_file)  # json_path에 전체 경로 설정
        try:
            # JSON 파일 열기 및 데이터 로드
            with open(json_path, 'r') as file:
                json_data = json.load(file)

            # 응답자 ID
            user_id = json_data['user']['R_id']

            # 스타일 선호도 확인
            img_name = json_data['item']['imgName']
            preference = json_data['item']['survey']['Q5']
            style_preferred = img_name if preference == 2 else ''
            style_not_preferred = img_name if preference == 1 else ''

            # 추출한 정보를 리스트에 추가
            data2.append({
                '응답자 ID': user_id,
                '스타일 선호': style_preferred,
                '스타일 비선호': style_not_preferred
            })
        except FileNotFoundError:
            print(f"File not found: {json_path}, skipping...")
            continue

# 데이터 프레임 생성
df2 = pd.DataFrame(data2)

# 중복된 ID에 대한 데이터 합치기
# '스타일 선호'와 '스타일 비선호' 열을 합치는 함수 정의
def combine_styles(series):
    # 리스트에서 빈 문자열을 제외하고 유니크한 값만 합칩니다.
    return ', '.join(set(filter(None, series)))

# 응답자 ID로 그룹화하고 스타일 데이터를 합치기
df2 = df2.groupby('응답자 ID').agg({
    '스타일 선호': combine_styles,
    '스타일 비선호': combine_styles
}).reset_index()


# 결과 출력
df2


import os
import json
import pandas as pd
#스타일 선호 정보표 (3/3)
#Training과 Validation 데이터프레임 합치기
# df와 df2를 '응답자 ID'를 기준으로 합치기
df_merged = pd.merge(df, df2, on='응답자 ID', how='outer')

# 인덱스 제거
df_merged.reset_index(drop=True, inplace=True)

# 멀티레벨 칼럼 생성
# 각 데이터프레임의 스타일 선호 및 비선호 칼럼 명을 정확히 확인 후 아래 명칭을 적절히 조정할 필요가 있음
df_merged.columns = pd.MultiIndex.from_tuples([
    ('', '응답자 ID'),
    ('Training', '스타일 선호'),
    ('Training', '스타일 비선호'),
    ('Validation', '스타일 선호'),
    ('Validation', '스타일 비선호')
])

# 상위 100명 응답자만 선택
df_top_100 = df_merged.head(100)

#행 레이블(인덱스) 숨기기
df_top_100.to_string(index=False)

# 결과 출력
df_top_100