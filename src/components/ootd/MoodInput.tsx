"use client";

import { Input } from "@/components/ui/input";
import { Sparkles } from "lucide-react";

interface MoodInputProps {
  value: string;
  onChange: (v: string) => void;
}

export default function MoodInput({ value, onChange }: MoodInputProps) {
  return (
    <div className="space-y-3">
      <label className="flex items-center gap-2 text-sm font-semibold text-foreground/80">
        <div className="rounded-lg bg-gradient-to-br from-blue-100 to-purple-100 p-1.5 border border-border/30">
          <Sparkles className="size-4 text-blue-600" />
        </div>
        오늘의 mood
      </label>
      <Input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="예: 소개팅 / 면접 / 학교 / 데이트 / 미니멀 / 러블리 / 스트릿…"
        className="bg-gradient-to-br from-amber-50/80 to-orange-50/60 border-amber-200/60 placeholder:text-amber-500/60 focus-visible:ring-amber-400/50 focus-visible:border-amber-300"
      />
    </div>
  );
}
