import Image from 'next/image';
import { SearchResult, DescriptorType } from '@/types/search';

interface ImageSearchCardProps {
  result: SearchResult;
  descriptorType: DescriptorType;
}

const ImageSearchCard: React.FC<ImageSearchCardProps> = ({ result, descriptorType }) => {
  return (
    <div className="border rounded-lg overflow-hidden shadow-md hover:shadow-xl transition-shadow">
      <div className="relative h-48 w-full">
        <Image 
          src={`http://localhost:5000/RSSCN7/${result.category}/${result.originalName}`} 
          alt={result.imagePath}
          fill
          className="object-cover"
        />
      </div>
      <div className="p-3">
        <h3 className="text-sm font-semibold truncate">{result.originalName}</h3>
        <p className="text-xs text-gray-600">Category: {result.category}</p>
        <p className="text-xs text-gray-500">Similarity: {(1 - result.distance).toFixed(6)}</p>
        <p className="text-xs text-gray-500">Distance: {(result.distance).toFixed(6)}</p>
        <p className="text-xs text-gray-500">Descriptor: {descriptorType}</p>
      </div>
    </div>
  );
};

export default ImageSearchCard;