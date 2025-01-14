// dashboard/transform-images/page.tsx
"use client";

import { useState, useEffect } from 'react';
import axios from 'axios';
import Image from 'next/image';
import { toast, Toaster } from 'react-hot-toast';
import SideBarAdmin from '@/components/SideBar';
import { 
  Crop, 
  Minimize2, 
  Rotate3D, 
  Scaling, 
  Move, 
  FlipHorizontal, 
  Check, 
  Plus, 
  Save 
} from 'lucide-react';

interface UploadedImage {
  filename: string;
  category: string;
  originalName: string;
}

type TransformationType = 'crop' | 'resize' | 'rotate' | 'scale' | 'translate' | 'flip';

interface Transformation {
  type: TransformationType;
  params: any;
}

const TransformImagesPage = () => {
  const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([]);
  const [selectedImage, setSelectedImage] = useState<UploadedImage | null>(null);
  const [transformationType, setTransformationType] = useState<TransformationType>('crop');
  const [appliedTransformations, setAppliedTransformations] = useState<Transformation[]>([]);
  const [currentTransformationParams, setCurrentTransformationParams] = useState({
    crop: { x: 0, y: 0, width: 0, height: 0 },
    resize: { width: 0, height: 0 },
    rotate: 0,
    scale: 1,
    translate: { x: 0, y: 0 },
    flip: 'horizontal'
  });

  // Fetch existing images (same as before)
  const fetchExistingImages = async () => {
    try {
      const response = await axios.get('http://localhost:5000/get-existing-images');
      setUploadedImages(response.data.images.map((image: any) => ({
        filename: image.filename,
        category: image.category,
        originalName: image.original_name
      })));
    } catch (error) {
      toast.error('Failed to load existing images');
    }
  };

  useEffect(() => {
    fetchExistingImages();
  }, []);

  const addTransformation = () => {
    const newTransformation: Transformation = {
      type: transformationType,
      params: currentTransformationParams[transformationType]
    };
    
    setAppliedTransformations(prev => [...prev, newTransformation]);
    toast.success(`${transformationType} transformation added`);
  };

  const handleTransform = async () => {
    if (!selectedImage || appliedTransformations.length === 0) {
      toast.error('Please select an image and add at least one transformation');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('image', selectedImage.filename);
      formData.append('category', selectedImage.category);
      formData.append('transformations', JSON.stringify(appliedTransformations));

      const response = await axios.post('http://localhost:5000/transform-image', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      toast.success('Image transformed successfully');
      fetchExistingImages();
      
      // Reset transformations after successful save
      setAppliedTransformations([]);
    } catch (error) {
      toast.error('Failed to transform image');
      console.error(error);
    }
  };

  const renderTransformationControls = () => {
    switch (transformationType) {
      case 'crop':
        return (
          <div className="grid grid-cols-2 gap-4">
            <input 
              type="number" 
              placeholder="X Coordinate" 
              value={currentTransformationParams.crop.x}
              onChange={(e) => setCurrentTransformationParams(prev => ({
                ...prev, 
                crop: { ...prev.crop, x: Number(e.target.value) }
              }))}
              className="input input-bordered w-full"
            />
            <input 
              type="number" 
              placeholder="Y Coordinate" 
              value={currentTransformationParams.crop.y}
              onChange={(e) => setCurrentTransformationParams(prev => ({
                ...prev, 
                crop: { ...prev.crop, y: Number(e.target.value) }
              }))}
              className="input input-bordered w-full"
            />
            <input 
              type="number" 
              placeholder="Width" 
              value={currentTransformationParams.crop.width}
              onChange={(e) => setCurrentTransformationParams(prev => ({
                ...prev, 
                crop: { ...prev.crop, width: Number(e.target.value) }
              }))}
              className="input input-bordered w-full"
            />
            <input 
              type="number" 
              placeholder="Height" 
              value={currentTransformationParams.crop.height}
              onChange={(e) => setCurrentTransformationParams(prev => ({
                ...prev, 
                crop: { ...prev.crop, height: Number(e.target.value) }
              }))}
              className="input input-bordered w-full"
            />
          </div>
        );
      case 'resize':
        return (
          <div className="grid grid-cols-2 gap-4">
            <input 
              type="number" 
              placeholder="Width" 
              value={currentTransformationParams.resize.width}
              onChange={(e) => setCurrentTransformationParams(prev => ({
                ...prev, 
                resize: { ...prev.resize, width: Number(e.target.value) }
              }))}
              className="input input-bordered w-full"
            />
            <input 
              type="number" 
              placeholder="Height" 
              value={currentTransformationParams.resize.height}
              onChange={(e) => setCurrentTransformationParams(prev => ({
                ...prev, 
                resize: { ...prev.resize, height: Number(e.target.value) }
              }))}
              className="input input-bordered w-full"
            />
          </div>
        );
      case 'rotate':
        return (
          <div>
            <input 
              type="range" 
              min="-180" 
              max="180" 
              value={currentTransformationParams.rotate}
              onChange={(e) => setCurrentTransformationParams(prev => ({
                ...prev, 
                rotate: Number(e.target.value)
              }))}
              className="range range-primary w-full"
            />
            <div className="w-full text-center mt-2">
              {currentTransformationParams.rotate}°
            </div>
          </div>
        );
      case 'scale':
        return (
          <div>
            <input 
              type="range" 
              min="0.1" 
              max="2" 
              step="0.1" 
              value={currentTransformationParams.scale}
              onChange={(e) => setCurrentTransformationParams(prev => ({
                ...prev, 
                scale: Number(e.target.value)
              }))}
              className="range range-primary w-full"
            />
            <div className="w-full text-center mt-2">
              Scale: {currentTransformationParams.scale}x
            </div>
          </div>
        );
      case 'translate':
        return (
          <div className="grid grid-cols-2 gap-4">
            <input 
              type="number" 
              placeholder="X Offset" 
              value={currentTransformationParams.translate.x}
              onChange={(e) => setCurrentTransformationParams(prev => ({
                ...prev, 
                translate: { ...prev.translate, x: Number(e.target.value) }
              }))}
              className="input input-bordered w-full"
            />
            <input 
              type="number" 
              placeholder="Y Offset" 
              value={currentTransformationParams.translate.y}
              onChange={(e) => setCurrentTransformationParams(prev => ({
                ...prev, 
                translate: { ...prev.translate, y: Number(e.target.value) }
              }))}
              className="input input-bordered w-full"
            />
          </div>
        );
      case 'flip':
        return (
          <div className="flex space-x-4">
            <button
              onClick={() => setCurrentTransformationParams(prev => ({
                ...prev, 
                flip: 'horizontal'
              }))}
              className={`
                btn ${currentTransformationParams.flip === 'horizontal' 
                  ? 'btn-primary' 
                  : 'btn-ghost'}
              `}
            >
              <FlipHorizontal /> Horizontal
            </button>
            <button
              onClick={() => setCurrentTransformationParams(prev => ({
                ...prev, 
                flip: 'vertical'
              }))}
              className={`
                btn ${currentTransformationParams.flip === 'vertical' 
                  ? 'btn-primary' 
                  : 'btn-ghost'}
              `}
            >
              <FlipHorizontal className="rotate-90" /> Vertical
            </button>
          </div>
        );
    }
  };

  return (
    <SideBarAdmin>
      <Toaster position="top-right" />
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-6">Image Transformation</h1>

        <div className="grid md:grid-cols-3 gap-8">
          {/* Image Selection */}
          <div className="bg-white shadow-lg rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">Select Image</h2>
            <div className="grid grid-cols-3 gap-4">
              {uploadedImages.map((image) => (
                <div 
                  key={image.filename}
                  onClick={() => setSelectedImage(image)}
                  className={`
                    relative cursor-pointer 
                    transform transition-all 
                    hover:scale-105 
                    ${selectedImage?.filename === image.filename ? 'ring-4 ring-blue-500' : ''}
                  `}
                >
                  <Image 
                    src={`http://localhost:5000/${image.category}/${image.filename}`}
                    alt={image.originalName}
                    width={100}
                    height={100}
                    className="object-cover rounded-md"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Transformation Selection */}
          <div className="bg-white shadow-lg rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">Choose Transformation</h2>
            <div className="grid grid-cols-2 gap-4">
              {[
                { type: 'crop', icon: <Crop /> },
                { type: 'resize', icon: <Minimize2 /> },
                { type: 'rotate', icon: <Rotate3D /> },
                { type: 'scale', icon: <Scaling /> },
                { type: 'translate', icon: <Move /> },
                { type: 'flip', icon: <FlipHorizontal /> }
              ].map(({ type, icon }) => (
                <button
                  key={type}
                  onClick={() => setTransformationType(type as TransformationType)}
                  className={`
                    flex items-center justify-center 
                    p-4 rounded-lg transition-all 
                    ${transformationType === type 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-100 hover:bg-gray-200'}
                  `}
                >
                  {icon}
                  <span className="ml-2 capitalize">{type}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Transformation Parameters */}
          <div className="bg-white shadow-lg rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">
              {transformationType.charAt(0).toUpperCase() + transformationType.slice(1)} Settings
            </h2>
            {renderTransformationControls()}
            
            {/* Add Transformation Button */}
            <button 
              onClick={addTransformation}
              disabled={!selectedImage}
              className="
                mt-4 w-full btn btn-secondary 
                flex items-center justify-center
                disabled:opacity-50 disabled:cursor-not-allowed
              "
            >
              <Plus className="mr-2" /> Add Transformation
            </button>
          </div>
        </div>

        {/* Applied Transformations */}
        {appliedTransformations.length > 0 && (
          <div className="mt-6 bg-white shadow-lg rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">Applied Transformations</h2>
            <div className="space-y-2">
              {appliedTransformations.map((transform, index) => (
                <div 
                  key={index} 
                  className="flex items-center justify-between bg-gray-100 p-3 rounded-lg"
                >
                  <div className="flex items-center">
                    <span className="font-medium capitalize mr-2">{transform.type}:</span>
                    <span className="text-gray-600">
                      {JSON.stringify(transform.params)}
                    </span>
                  </div>
                  <button 
                    onClick={() => {
                      // Remove transformation at this index
                      setAppliedTransformations(prev => 
                        prev.filter((_, i) => i !== index)
                      );
                    }}
                    className="btn btn-xs btn-error"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Transform and Save Buttons */}
        <div className="mt-6 flex space-x-4">
          <button 
            onClick={handleTransform}
            disabled={!selectedImage || appliedTransformations.length === 0}
            className="
              flex-1 btn btn-primary 
              disabled:opacity-50 disabled:cursor-not-allowed
              flex items-center justify-center
            "
          >
            <Check className="mr-2" /> Apply Transformations
          </button>
        </div>

        {/* Preview Area */}
        {selectedImage && (
          <div className="mt-8 bg-white shadow-lg rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">Preview</h2>
            <div className="flex justify-center">
              <Image 
                src={`http://localhost:5000/${selectedImage.category}/${selectedImage.filename}`}
                alt={selectedImage.originalName}
                width={500}
                height={500}
                className="object-contain rounded-md"
              />
            </div>
          </div>
        )}
      </div>
    </SideBarAdmin>
  );
};

export default TransformImagesPage;