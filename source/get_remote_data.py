import gdown

file_id = "1K4JlDqdso9-wwOacR3hj4uVP0Hxxt2L1"  # replace with your file ID
url = f"https://drive.google.com/uc?id={file_id}"
output = "../data/processed_audio/128f_chunked_spectrograms.tar.gz"  # desired output file name

gdown.download(url, output, quiet=False)

#flattened_npy_sectrograms.tar.gz
#https://drive.google.com/file/d/1gz7eJPB-pDTN7qee0ZoEOCUaQ-TLnRqV/view?usp=drive_link

#128 frame chunks
#https://drive.google.com/file/d/1K4JlDqdso9-wwOacR3hj4uVP0Hxxt2L1/view?usp=drive_link