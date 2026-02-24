"use client";

import { useState, useMemo } from "react";
import { Shirt } from "lucide-react";
import { type ClosetItemView, getCategoryLabel } from "@/lib/types/closet-view";
import ClosetItemCard from "./ClosetItemCard";
import ClosetItemDialog from "./ClosetItemDialog";

const CATEGORY_FILTERS = [
  { key: "all", label: "전체" },
  { key: "top", label: "상의" },
  { key: "bottom", label: "하의" },
  { key: "outer", label: "아우터" },
  { key: "dress", label: "원피스" },
] as const;

const PAGE_SIZE = 30;

interface ClosetGridProps {
  items: ClosetItemView[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  highlightId?: string | null;
}

export default function ClosetGrid({
  items,
  selectedId,
  onSelect,
  highlightId,
}: ClosetGridProps) {
  const [dialogItem, setDialogItem] = useState<ClosetItemView | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  const filtered = useMemo(() => {
    if (activeCategory === "all") return items;
    return items.filter((item) => item.category === activeCategory);
  }, [items, activeCategory]);

  const visible = filtered.slice(0, visibleCount);
  const hasMore = visibleCount < filtered.length;

  function handleClick(item: ClosetItemView) {
    onSelect(item.id);
    setDialogItem(item);
    setDialogOpen(true);
  }

  function handleCategoryChange(key: string) {
    setActiveCategory(key);
    setVisibleCount(PAGE_SIZE);
  }

  // 카테고리별 개수
  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = { all: items.length };
    for (const item of items) {
      counts[item.category] = (counts[item.category] || 0) + 1;
    }
    return counts;
  }, [items]);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 rounded-xl bg-gradient-to-r from-slate-50/50 to-blue-50/30 p-3 border border-border/30">
        <div className="rounded-lg bg-gradient-to-br from-blue-100 to-purple-100 p-1.5 border border-border/30">
          <Shirt className="size-4 text-blue-600" />
        </div>
        <h3 className="text-sm font-semibold text-foreground/80">
          내 옷장
        </h3>
        <span className="text-xs text-muted-foreground bg-white/60 rounded-md px-2 py-0.5 border border-border/30">
          {items.length}벌
        </span>
      </div>

      {/* 카테고리 필터 탭 */}
      <div className="flex gap-2 flex-wrap">
        {CATEGORY_FILTERS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => handleCategoryChange(key)}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${
              activeCategory === key
                ? "bg-primary text-primary-foreground border-primary shadow-sm"
                : "bg-white text-muted-foreground border-border/40 hover:border-border hover:bg-slate-50"
            }`}
          >
            {label}
            <span className="ml-1 opacity-70">
              {categoryCounts[key] || 0}
            </span>
          </button>
        ))}
      </div>

      {/* 아이템 그리드 */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        {visible.map((item) => (
          <ClosetItemCard
            key={item.id}
            item={item}
            selected={selectedId === item.id}
            highlighted={highlightId === item.id}
            onClick={() => handleClick(item)}
          />
        ))}
      </div>

      {/* 더보기 버튼 */}
      {hasMore && (
        <button
          onClick={() => setVisibleCount((prev) => prev + PAGE_SIZE)}
          className="w-full py-2.5 text-sm text-muted-foreground hover:text-foreground bg-white hover:bg-slate-50 border border-border/40 rounded-xl transition-all"
        >
          더보기 ({filtered.length - visibleCount}개 남음)
        </button>
      )}

      <ClosetItemDialog
        item={dialogItem}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
      />
    </div>
  );
}
