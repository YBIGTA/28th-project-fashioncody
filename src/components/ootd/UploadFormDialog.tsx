"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

const CATEGORIES = [
  { key: "top", label: "상의" },
  { key: "bottom", label: "하의" },
  { key: "outer", label: "아우터" },
  { key: "dress", label: "원피스" },
] as const;

export interface UploadFormData {
  category: string;
  color: string;
  subType: string;
}

interface UploadFormDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  previewUrl: string;
  loading: boolean;
  onSubmit: (data: UploadFormData) => void;
}

export default function UploadFormDialog({
  open,
  onOpenChange,
  previewUrl,
  loading,
  onSubmit,
}: UploadFormDialogProps) {
  const [category, setCategory] = useState("top");
  const [color, setColor] = useState("");
  const [subType, setSubType] = useState("");

  function handleSubmit() {
    if (!color.trim()) return;
    onSubmit({ category, color: color.trim(), subType: subType.trim() });
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>옷장에 추가</DialogTitle>
          <DialogDescription>
            아이템 정보를 입력해주세요.
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-[140px_1fr] gap-5">
          {/* 이미지 미리보기 */}
          <div className="aspect-[4/5] overflow-hidden rounded-xl border border-border/60">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            {previewUrl && (
              <img
                src={previewUrl}
                alt="업로드 미리보기"
                className="h-full w-full object-cover"
              />
            )}
          </div>

          {/* 입력 폼 */}
          <div className="space-y-4">
            {/* 카테고리 */}
            <div className="space-y-1.5">
              <label className="text-sm font-medium">
                카테고리 <span className="text-red-500">*</span>
              </label>
              <div className="flex flex-wrap gap-1.5">
                {CATEGORIES.map(({ key, label }) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setCategory(key)}
                    className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${
                      category === key
                        ? "bg-primary text-primary-foreground border-primary shadow-sm"
                        : "bg-white text-muted-foreground border-border/40 hover:border-border hover:bg-slate-50"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* 색상 */}
            <div className="space-y-1.5">
              <label className="text-sm font-medium">
                색상 <span className="text-red-500">*</span>
              </label>
              <Input
                value={color}
                onChange={(e) => setColor(e.target.value)}
                placeholder="예: 화이트, 블랙, 네이비"
                className="text-sm rounded-lg"
              />
            </div>

            {/* 유형 */}
            <div className="space-y-1.5">
              <label className="text-sm font-medium">유형</label>
              <Input
                value={subType}
                onChange={(e) => setSubType(e.target.value)}
                placeholder="예: 니트, 셔츠, 슬랙스"
                className="text-sm rounded-lg"
              />
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button
            onClick={handleSubmit}
            disabled={!color.trim() || loading}
            className="w-full gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
          >
            {loading ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                등록 중...
              </>
            ) : (
              "등록하기"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
