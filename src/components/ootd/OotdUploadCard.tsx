"use client";

import { useRef } from "react";
import { Camera, X, ImageIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

interface OotdUploadCardProps {
  previewUrl: string;
  onUpload: (file: File) => void;
  onRemove: () => void;
}

export default function OotdUploadCard({
  previewUrl,
  onUpload,
  onRemove,
}: OotdUploadCardProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold tracking-wide text-foreground/80 uppercase">
          OOTD
        </h3>
        <Button
          variant="outline"
          size="sm"
          className="gap-1.5 text-xs rounded-lg"
          onClick={() => inputRef.current?.click()}
        >
          <Camera className="size-3.5" />
          업로드
        </Button>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleChange}
        />
      </div>

      <div className="relative aspect-[3/4] w-full overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-muted/40 to-muted/20 notion-hover">
        {previewUrl ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="OOTD preview"
              className="h-full w-full object-cover transition-transform duration-300 hover:scale-105"
            />
            <button
              onClick={onRemove}
              className="absolute top-3 right-3 rounded-full bg-black/60 backdrop-blur-sm p-2 text-white transition-all duration-200 hover:bg-black/80 hover:scale-110 shadow-lg"
            >
              <X className="size-4" />
            </button>
          </>
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-4 text-muted-foreground">
            <div className="rounded-2xl bg-gradient-to-br from-blue-100/50 to-purple-100/50 p-6 border border-border/30">
              <ImageIcon className="size-12 opacity-40" />
            </div>
            <p className="text-sm font-medium">오늘의 코디를 올려보세요</p>
          </div>
        )}
      </div>
    </div>
  );
}
