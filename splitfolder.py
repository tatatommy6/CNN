import splitfolders

# 데이터셋이 위치한 경로
input_folder = '/Users/kimminkyeol/Desktop/PetImages'

# 데이터를 분할하여 저장할 경로
output_folder = '/Users/kimminkyeol/Desktop/output'

# 데이터 분할 비율 설정: 80% 훈련, 10% 검증, 10% 테스트
splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.8, .1, .1))
