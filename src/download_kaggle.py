import kagglehub

# Download latest version
path = kagglehub.competition_download('dogs-vs-cats')

print("Path to competition files:", path)