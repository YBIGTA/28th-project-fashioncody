"use client";

import { Badge } from "@/components/ui/badge";
import { type ClosetItemView, getCategoryLabel } from "@/lib/types/closet-view";

interface ClosetItemCardProps {
  item: ClosetItemView;
  selected: boolean;
  highlighted?: boolean;
  onClick: () => void;
}

export default function ClosetItemCard({
  item,
  selected,
  highlighted,
  onClick,
}: ClosetItemCardProps) {
  return (
    <button
      onClick={onClick}
      className={`group relative overflow-hidden rounded-2xl bg-white text-left transition-all duration-300 ease-out notion-hover ${
        highlighted
          ? "ring-2 ring-green-400 shadow-lg shadow-green-200/40 scale-[1.02] animate-pulse"
          : selected
            ? "ring-2 ring-primary shadow-lg shadow-primary/20 scale-[1.02]"
            : "ring-1 ring-border/60 hover:ring-border hover:shadow-md"
      }`}
    >
      <div className="aspect-[4/5] w-full overflow-hidden rounded-t-2xl">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={item.imageUrl}
          alt={item.name}
          className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-110"
        />
      </div>

      <div className="space-y-1.5 p-3 bg-gradient-to-b from-white to-slate-50/50">
        <p className="truncate text-sm font-medium text-foreground/90">
          {item.name}
        </p>
        <div className="flex flex-wrap gap-1.5">
          <Badge variant="secondary" className="text-[10px] px-2 py-0.5 rounded-md">
            {getCategoryLabel(item.category)}
          </Badge>
          {item.tags?.slice(0, 2).map((tag) => (
            <Badge
              key={tag}
              variant="outline"
              className="text-[10px] px-2 py-0.5 rounded-md text-muted-foreground border-border/40"
            >
              {tag}
            </Badge>
          ))}
        </div>
      </div>

      {selected && (
        <div className="absolute top-3 right-3 flex size-6 items-center justify-center rounded-full bg-primary text-xs text-primary-foreground shadow-lg ring-2 ring-primary/20 animate-in zoom-in-50 duration-200">
          ✓
        </div>
      )}
    </button>
  );
}
