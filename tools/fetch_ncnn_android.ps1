$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$version = "20260526"
$archive = Join-Path $env:TEMP "ncnn-$version-android-vulkan-shared.zip"
$extract = Join-Path $env:TEMP "ncnn-$version-android-vulkan-shared"
$url = "https://github.com/Tencent/ncnn/releases/download/$version/ncnn-$version-android-vulkan-shared.zip"
$source = Join-Path $extract "ncnn-$version-android-vulkan-shared\arm64-v8a"
$cpp = Join-Path $root "android\app\src\main\cpp\ncnn"
$libs = Join-Path $root "android\app\src\main\jniLibs\arm64-v8a"

Invoke-WebRequest -Uri $url -OutFile $archive
Expand-Archive -LiteralPath $archive -DestinationPath $extract -Force
New-Item -ItemType Directory -Force -Path $cpp, $libs | Out-Null
Copy-Item -LiteralPath (Join-Path $source "include\ncnn") -Destination $cpp -Recurse -Force
Copy-Item -LiteralPath (Join-Path $source "lib\libncnn.so") -Destination (Join-Path $libs "libncnn.so") -Force
