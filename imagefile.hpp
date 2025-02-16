#define _CRT_SECURE_NO_WARNINGS
#include <dirent.h>
#include <string>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iomanip>
#include "exif.h"

std::vector<std::string> getFilesInFolder(const std::string& folderPath){
    std::vector<std::string> fileNames;

    DIR* dir;
    struct  dirent* ent;
    
    if((dir = opendir(folderPath.c_str())) != NULL){
        while((ent = readdir(dir)) != NULL){
            std::string fileName = ent->d_name;
            if(fileName != "." && fileName != ".."){
                fileNames.push_back(folderPath + "/" + fileName);
            }
        }
        closedir(dir);
    }
    
    return fileNames;
}

std::vector<std::string> getImgList(const std::string& folderPath){
    std::vector<std::string> fileNames = getFilesInFolder(folderPath);

    std::sort(fileNames.begin(), fileNames.end());

    return fileNames;
}

void wirteImagesPathsToFile(const std::string& folderPath, const std::string& outputFileName){
    std::vector<std::string> fileNames = getFilesInFolder(folderPath);

    std::sort(fileNames.begin(), fileNames.end());

    std::ofstream outputFile(outputFileName);

    if(!outputFile.is_open()){
        std::cerr << "can not open the outputFile!" << std::endl;
        return;
    }

    for(const std::string& fileName : fileNames){
        outputFile << fileName << std::endl;
    }

    outputFile.close();
}

void EXIT(const char* buf)
{
	std::cout << buf << "\n";
	getchar();
}

std::pair<double, double> get_image_info(std::string jpg_path)
{
    std::pair<double, double> geolocation;
	const char* JpgPath = jpg_path.c_str();
	FILE* fp = fopen(JpgPath, "rb");
	if (!fp) { 
        EXIT("read jpg fail!..");
        return geolocation; 
    }
	if (fp == NULL) return geolocation;

	fseek(fp, 0, SEEK_END);
	unsigned long fsize = ftell(fp);
	rewind(fp);
	unsigned char* buf = new unsigned char[fsize];
	if (fread(buf, 1, fsize, fp) != fsize)
	{
		EXIT("Can't read file.");
		delete[] buf;
	}


	easyexif::EXIFInfo result;
	int code = result.parseFrom(buf, fsize);
	delete[] buf;
	if (code)
	{

		EXIT("Error parsing EXIF error");
	}

    geolocation = {result.GeoLocation.Latitude, result.GeoLocation.Longitude};

	FILE* file = fopen("../GeoLocation.txt","a+");
	if (file == NULL) {
		return geolocation;
	}
	fclose(fp);
	// cout << std::fixed << setprecision(15) << result.GeoLocation.Latitude << '\t' << result.GeoLocation.Longitude << endl;
	// fprintf(file, "%f,%f\n", result.GeoLocation.Latitude, result.GeoLocation.Longitude);


	// printf("Camera make          : %s\n", result.Make.c_str());						
	// printf("Camera model         : %s\n", result.Model.c_str());					
	// printf("Software             : %s\n", result.Software.c_str());					
	// printf("Bits per sample      : %d\n", result.BitsPerSample);					
	// printf("Image width          : %d\n", result.ImageWidth);						
	// printf("Image height         : %d\n", result.ImageHeight);						
	// printf("Image description    : %s\n", result.ImageDescription.c_str());			
	// printf("Image orientation    : %d\n", result.Orientation);						
	// printf("Image copyright      : %s\n", result.Copyright.c_str());				
	// printf("Image date/time      : %s\n", result.DateTime.c_str());					
	// printf("Original date/time   : %s\n", result.DateTimeOriginal.c_str());			
	// printf("Digitize date/time   : %s\n", result.DateTimeDigitized.c_str());		
	// printf("Subsecond time       : %s\n", result.SubSecTimeOriginal.c_str());		
	// printf("Exposure time        : 1/%d s\n",
	// 	(unsigned)(1.0 / result.ExposureTime));										
	// printf("F-stop               : f/%.1f\n", result.FNumber);						
	// printf("Exposure program     : %d\n", result.ExposureProgram);					
	// printf("ISO speed            : %d\n", result.ISOSpeedRatings);					
	// printf("Subject distance     : %f m\n", result.SubjectDistance);				
	// printf("Exposure bias        : %f EV\n", result.ExposureBiasValue);				
	// printf("Flash used?          : %d\n", result.Flash);							
	// printf("Flash returned light : %d\n", result.FlashReturnedLight);				
	// printf("Flash mode           : %d\n", result.FlashMode);						
	// printf("Metering mode        : %d\n", result.MeteringMode);						
	// printf("Lens focal length    : %f mm\n", result.FocalLength);					
	// printf("35mm focal length    : %u mm\n", result.FocalLengthIn35mm);				
	// fprintf(file, "%s:\n", jpg_path.c_str());
    // fprintf(file, "GPS Latitude         : %f deg (%f deg, %f min, %f sec %c)\n",
	// 	result.GeoLocation.Latitude,												
	// 	result.GeoLocation.LatComponents.degrees,									
	// 	result.GeoLocation.LatComponents.minutes,									
	// 	result.GeoLocation.LatComponents.seconds,									
	// 	result.GeoLocation.LatComponents.direction);								
	// fprintf(file, "GPS Longitude        : %f deg (%f deg, %f min, %f sec %c)\n",
	// 	result.GeoLocation.Longitude,												
	// 	result.GeoLocation.LonComponents.degrees,									
	// 	result.GeoLocation.LonComponents.minutes,									
	// 	result.GeoLocation.LonComponents.seconds,									
	// 	result.GeoLocation.LonComponents.direction);								
	// printf("GPS Altitude         : %f m\n", result.GeoLocation.Altitude);			
	// printf("GPS Precision (DOP)  : %f\n", result.GeoLocation.DOP);					
	// printf("Lens min focal length: %f mm\n", result.LensInfo.FocalLengthMin);		
	// printf("Lens max focal length: %f mm\n", result.LensInfo.FocalLengthMax);		
	// printf("Lens f-stop min      : f/%.1f\n", result.LensInfo.FStopMin);			
	// printf("Lens f-stop max      : f/%.1f\n", result.LensInfo.FStopMax);			
	// printf("Lens make            : %s\n", result.LensInfo.Make.c_str());			
	// printf("Lens model           : %s\n", result.LensInfo.Model.c_str());			
	// printf("Focal plane XRes     : %f\n", result.LensInfo.FocalPlaneXResolution);	
	// printf("Focal plane YRes     : %f\n", result.LensInfo.FocalPlaneYResolution);	
	
    return geolocation;
}

static easyexif::EXIFInfo getCameraInfo(std::string jpg_path){
	easyexif::EXIFInfo info;
	const char* JpgPath = jpg_path.c_str();
	FILE* fp = fopen(JpgPath, "rb");
	if (!fp) { 
        EXIT("read jpg fail!..");
        return info; 
    }
	if (fp == NULL) return info;

	fseek(fp, 0, SEEK_END);
	unsigned long fsize = ftell(fp);
	rewind(fp);
	unsigned char* buf = new unsigned char[fsize];
	if (fread(buf, 1, fsize, fp) != fsize)
	{
		EXIT("Can't read file.");
		delete[] buf;
	}

	int code = info.parseFrom(buf, fsize);
	delete[] buf;
	if (code)
	{
		EXIT("Error parsing EXIF error");
	}

	fclose(fp);
	return info;
}

void GetGeoLocation(std::string path){
	std::vector<std::string> filename = getFilesInFolder(path);
    std::sort(filename.begin(),filename.end());
    easyexif::EXIFInfo exifinfo;
    std::vector<std::string>::iterator itbegin = filename.begin();
	std::vector<std::string>::iterator itend = filename.end();
    std::vector<std::pair<double, double>> geolocations; 
	while (itbegin != itend)
	{
		geolocations.push_back(get_image_info(*itbegin));
		std::cout << *itbegin << std::endl;
		itbegin++;
	}
    // for(int i = 0; i < geolocations.size(); i++){
    //     std::cout << geolocations[i].first << "|" << geolocations[i].second << std::endl;
    // }
    // for(int i = 1; i < geolocations.size(); i++){
    //     std::cout << geolocations[i].second - geolocations[i - 1].second<< std::endl;
    // }
}