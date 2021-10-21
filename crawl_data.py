from simple_image_download import simple_image_download as simp

name = ['peach blossom', 'apricot blossom']

response = simp.simple_image_download()

for i in name:
    response.download(i, 10)