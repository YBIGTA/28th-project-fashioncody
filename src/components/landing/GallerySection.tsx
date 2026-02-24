"use client";

import { type ClosetItemView } from "@/lib/types/closet-view";
import ClosetItemCard from "@/components/ootd/ClosetItemCard";
import ClosetItemDialog from "@/components/ootd/ClosetItemDialog";
import { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";

export default function GallerySection() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [dialogItem, setDialogItem] = useState<ClosetItemView | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [galleryItems, setGalleryItems] = useState<ClosetItemView[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchItems() {
      try {
        const res = await fetch("/api/closet");
        if (!res.ok) throw new Error("Failed to fetch");
        const data = await res.json();

        const views: ClosetItemView[] = data.items
          .slice(0, 9)
          .map((item: any) => ({
            id: item.id,
            name:
              item.name ||
              item.attributes?.sub_type ||
              item.attributes?.category ||
              "Unknown",
            imageUrl: item.imageUrl,
            category: item.attributes?.category || "top",
            color: item.attributes?.color,
            subType: item.attributes?.sub_type,
            material: item.attributes?.material?.[0]?.value,
            fit: item.attributes?.fit,
            season: item.season,
            tags: item.tags,
            createdAt: item.createdAt,
          }));

        setGalleryItems(views);
      } catch {
        // Gallery is non-critical, fail silently
      } finally {
        setLoading(false);
      }
    }
    fetchItems();
  }, []);

  function handleClick(item: ClosetItemView) {
    setSelectedId(item.id);
    setDialogItem(item);
    setDialogOpen(true);
  }

  return (
    <section
      id="gallery"
      className="py-24 bg-white"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">
            인기 코디 갤러리
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            다른 사용자들이 만든 스타일을 둘러보고 영감을 얻어보세요
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="size-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {galleryItems.map((item) => (
              <ClosetItemCard
                key={item.id}
                item={item}
                selected={selectedId === item.id}
                onClick={() => handleClick(item)}
              />
            ))}
          </div>
        )}

        <div className="text-center mt-12">
          <button className="text-primary font-semibold hover:underline transition-colors">
            더 많은 코디 보기 →
          </button>
        </div>
      </div>

      <ClosetItemDialog
        item={dialogItem}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
      />
    </section>
  );
}
