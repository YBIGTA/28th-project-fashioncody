"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import {
  type ClosetItemView,
  getCategoryLabel,
  getSeasonLabel,
} from "@/lib/types/closet-view";

interface ClosetItemDialogProps {
  item: ClosetItemView | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function ClosetItemDialog({
  item,
  open,
  onOpenChange,
}: ClosetItemDialogProps) {
  if (!item) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md p-0 overflow-hidden">
        <div className="aspect-[3/4] w-full overflow-hidden rounded-t-2xl">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={item.imageUrl}
            alt={item.name}
            className="h-full w-full object-cover"
          />
        </div>

        <div className="space-y-4 px-6 pb-6 pt-4 bg-gradient-to-b from-white to-slate-50/50">
          <DialogHeader>
            <DialogTitle className="text-xl font-semibold">{item.name}</DialogTitle>
          </DialogHeader>

          <div className="grid grid-cols-2 gap-y-3 text-sm">
            <span className="text-muted-foreground font-medium">카테고리</span>
            <span className="font-semibold text-foreground">
              {getCategoryLabel(item.category)}
            </span>

            {item.color && (
              <>
                <span className="text-muted-foreground font-medium">컬러</span>
                <span className="font-semibold text-foreground">{item.color}</span>
              </>
            )}

            {item.subType && (
              <>
                <span className="text-muted-foreground font-medium">유형</span>
                <span className="font-semibold text-foreground">{item.subType}</span>
              </>
            )}

            {item.material && (
              <>
                <span className="text-muted-foreground font-medium">소재</span>
                <span className="font-semibold text-foreground">{item.material}</span>
              </>
            )}

            {item.fit && (
              <>
                <span className="text-muted-foreground font-medium">핏</span>
                <span className="font-semibold text-foreground">{item.fit}</span>
              </>
            )}

            {item.season && item.season.length > 0 && (
              <>
                <span className="text-muted-foreground font-medium">시즌</span>
                <div className="flex flex-wrap gap-1.5">
                  {item.season.map((s) => (
                    <Badge key={s} variant="secondary" className="text-xs rounded-md">
                      {getSeasonLabel(s)}
                    </Badge>
                  ))}
                </div>
              </>
            )}

            <span className="text-muted-foreground font-medium">등록일</span>
            <span className="font-semibold text-foreground">{item.createdAt}</span>
          </div>

          {item.tags && item.tags.length > 0 && (
            <div className="flex flex-wrap gap-2 pt-2 border-t border-border/40">
              {item.tags.map((tag) => (
                <Badge key={tag} variant="outline" className="text-xs rounded-md border-border/40">
                  #{tag}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
