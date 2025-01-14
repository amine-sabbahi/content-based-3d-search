import Image from 'next/image';
import { cn } from '@/lib/utils';

interface ImageSearchCardProps {
  imagePath: string;
  originalName?: string;
  distance?: number;
  category?: string;
  onRelevant?: () => void;
  onIrrelevant?: () => void;
  isRelevant?: boolean;
  isIrrelevant?: boolean;
  className?: string;
  
}

const ImageSearchCard: React.FC<ImageSearchCardProps> = ({
  imagePath, 
  originalName, 
  distance,
  category,
  onRelevant,
  onIrrelevant,
  isRelevant = false,
  isIrrelevant = false,
  className
  
}) => {
  return (
    <div className={cn(
      "border rounded-lg shadow-sm overflow-hidden hover:shadow-md transition-all duration-300",
      className
    )}>
      {/* Image Display */}
      <div className="relative w-full h-48 overflow-hidden">
        <Image 
          src={`http://localhost:5000/RSSCN7/${category}/${originalName}`}
          alt={originalName || 'Search Result'}
          fill
          className="object-cover"
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
        />
      </div>
      

      {/* Card Content */}
      <div className="p-4">
        {/* Buttons Container */}
        <div className="flex space-x-2 mb-2">
          <button
            onClick={onRelevant}
            className={cn(
              "flex-1 py-2 rounded transition-colors duration-300",
              isRelevant 
                ? "bg-green-500 text-white" 
                : "bg-gray-100 text-gray-700 hover:bg-green-100"
            )}
          >
            Relevant
          </button>
          <button
            onClick={onIrrelevant}
            className={cn(
              "flex-1 py-2 rounded transition-colors duration-300",
              isIrrelevant 
                ? "bg-red-500 text-white" 
                : "bg-gray-100 text-gray-700 hover:bg-red-100"
            )}
          >
            Irrelevant
          </button>
        </div>
        {originalName !== undefined && (
          <div className="text-sm text-gray-600 text-center">
            Image Name: {originalName}
          </div>
        )}
        {category !== undefined && (
          <div className="text-sm text-gray-600 text-center">
            Category: {category}
          </div>
        )}

        {/* Distance Information */}
        {distance !== undefined && (
          <div className="text-sm text-gray-600 text-center">
            
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageSearchCard;