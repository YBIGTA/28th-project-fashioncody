"use client";

import { useState, useMemo } from "react";
import { Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { ClosetItemView } from "@/lib/types/closet-view";

const CATEGORY_TABS = [
  { key: "top", label: "상의", color: "bg-blue-600/80" },
  { key: "bottom", label: "하의", color: "bg-purple-600/80" },
  { key: "dress", label: "원피스", color: "bg-pink-600/80" },
  { key: "outer", label: "아우터", color: "bg-orange-600/80" },
] as const;

type CategoryKey = (typeof CATEGORY_TABS)[number]["key"];

interface SelectedItemsGridProps {
  items: {
    top: ClosetItemView[];
    bottom: ClosetItemView[];
    dress: ClosetItemView[];
    outer: ClosetItemView[];
  };
}

export default function SelectedItemsGrid({ items }: SelectedItemsGridProps) {
  const [activeTab, setActiveTab] = useState<CategoryKey>("top");

  const totalCount = useMemo(
    () => items.top.length + items.bottom.length + items.dress.length + items.outer.length,
    [items]
  );

  const currentItems = items[activeTab];

  if (totalCount === 0) return null;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Sparkles className="size-5 text-amber-500" />
          AI가 선정한 상황 적합 아이템
        </h3>
        <Badge variant="secondary" className="rounded-lg">
          총 {totalCount}벌
        </Badge>
      </div>

      {/* Category tabs */}
      <div className="flex gap-2 flex-wrap">
        {CATEGORY_TABS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${
              activeTab === key
                ? "bg-primary text-primary-foreground border-primary shadow-sm"
                : "bg-white text-muted-foreground border-border/40 hover:border-border hover:bg-slate-50"
            }`}
          >
            {label}
            <span className="ml-1 opacity-70">{items[key].length}</span>
          </button>
        ))}
      </div>

      {/* Items grid */}
      {currentItems.length > 0 ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-5">
          {currentItems.map((item) => {
            const tabMeta = CATEGORY_TABS.find((t) => t.key === activeTab);
            return (
              <div
                key={item.id}
                className="group relative rounded-xl overflow-hidden border border-border/40 bg-white shadow-sm hover:shadow-md transition-all"
              >
                <div className="aspect-[4/5] overflow-hidden">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={item.imageUrl}
                    alt={item.name}
                    className="h-full w-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <div
                    className={`absolute top-1.5 left-1.5 ${tabMeta?.color || "bg-gray-600/80"} text-white text-[10px] px-1.5 py-0.5 rounded-md font-medium`}
                  >
                    {tabMeta?.label}
                  </div>
                </div>
                <div className="p-2 space-y-0.5">
                  <p className="text-xs font-medium truncate">{item.name}</p>
                  {item.color && (
                    <p className="text-[10px] text-muted-foreground truncate">
                      {item.color}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground text-center py-6">
          해당 카테고리의 선정 아이템이 없습니다.
        </p>
      )}
    </div>
  );
}
