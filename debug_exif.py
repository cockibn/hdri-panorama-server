#!/usr/bin/env python3
"""
Debug EXIF data to see what's actually available.
"""

import piexif
import os

def debug_exif_data(image_path):
    """Debug all available EXIF data."""
    print(f"🔍 EXIF Debug for: {image_path}")
    print("=" * 60)
    
    try:
        exif_dict = piexif.load(image_path)
        
        for ifd_name in ["0th", "Exif", "GPS", "1st"]:
            if ifd_name in exif_dict and exif_dict[ifd_name]:
                print(f"\n📊 {ifd_name} IFD:")
                for tag_id, value in exif_dict[ifd_name].items():
                    try:
                        # Get tag name if possible
                        if ifd_name == "0th":
                            tag_name = piexif.ImageIFD.__dict__.get(str(tag_id), f"Unknown_{tag_id}")
                        elif ifd_name == "Exif":
                            tag_name = piexif.ExifIFD.__dict__.get(str(tag_id), f"Unknown_{tag_id}")
                        elif ifd_name == "GPS":
                            tag_name = piexif.GPSIFD.__dict__.get(str(tag_id), f"Unknown_{tag_id}")
                        else:
                            tag_name = f"Tag_{tag_id}"
                            
                        # Get readable tag name
                        for attr_name in dir(piexif.ExifIFD):
                            if hasattr(piexif.ExifIFD, attr_name) and getattr(piexif.ExifIFD, attr_name) == tag_id:
                                tag_name = attr_name
                                break
                                
                        print(f"   {tag_name} ({tag_id}): {value}")
                    except:
                        print(f"   Tag {tag_id}: {value}")
                        
        # Check specifically for the fields we're looking for
        print(f"\n🎯 Key Fields Check:")
        if "Exif" in exif_dict:
            exif_data = exif_dict["Exif"]
            
            focal_length = exif_data.get(piexif.ExifIFD.FocalLength)
            print(f"   FocalLength: {focal_length}")
            
            focal_plane_x_res = exif_data.get(piexif.ExifIFD.FocalPlaneXResolution)
            focal_plane_y_res = exif_data.get(piexif.ExifIFD.FocalPlaneYResolution)
            focal_plane_unit = exif_data.get(piexif.ExifIFD.FocalPlaneResolutionUnit)
            
            print(f"   FocalPlaneXResolution: {focal_plane_x_res}")
            print(f"   FocalPlaneYResolution: {focal_plane_y_res}")
            print(f"   FocalPlaneResolutionUnit: {focal_plane_unit}")
            
            iso = exif_data.get(piexif.ExifIFD.ISOSpeedRatings)
            aperture = exif_data.get(piexif.ExifIFD.FNumber)
            exposure = exif_data.get(piexif.ExifIFD.ExposureTime)
            wb = exif_data.get(piexif.ExifIFD.WhiteBalance)
            
            print(f"   ISOSpeedRatings: {iso}")
            print(f"   FNumber (Aperture): {aperture}")
            print(f"   ExposureTime: {exposure}")
            print(f"   WhiteBalance: {wb}")
        
        if "0th" in exif_dict:
            img_width = exif_dict["0th"].get(piexif.ImageIFD.ImageWidth)
            img_height = exif_dict["0th"].get(piexif.ImageIFD.ImageLength)
            make = exif_dict["0th"].get(piexif.ImageIFD.Make)
            model = exif_dict["0th"].get(piexif.ImageIFD.Model)
            
            print(f"   ImageWidth: {img_width}")
            print(f"   ImageLength: {img_height}")
            print(f"   Make: {make}")
            print(f"   Model: {model}")
            
    except Exception as e:
        print(f"❌ Error reading EXIF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    image_path = "/Users/gianluca/Downloads/IMG_7021.jpg"
    if os.path.exists(image_path):
        debug_exif_data(image_path)
    else:
        print(f"❌ Image not found: {image_path}")