import wget
import zipfile
def main():
    url = 'https://s3-eu-west-1.amazonaws.com/recoordio-zoo/dataset/synthetic2_128x128.zip'
    file = wget.download(url)
    zips = zipfile.ZipFile(r'./synthetic2_128x128.zip')
    zips.extractall(r'./')  

if __name__ == '__main__':
    main()