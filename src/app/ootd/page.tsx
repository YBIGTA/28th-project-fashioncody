"use client";

import { useState, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { RotateCcw, Wand2, Loader2, Plus } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

import OotdHeader from "@/components/ootd/OotdHeader";
import DateWeatherHeader from "@/components/ootd/DateWeatherHeader";
import OotdUploadCard from "@/components/ootd/OotdUploadCard";
import MoodInput from "@/components/ootd/MoodInput";
import ClosetGrid from "@/components/ootd/ClosetGrid";
import SelectedItemsGrid from "@/components/ootd/SelectedItemsGrid";
import RecommendationResult, {
  type RecommendationItem,
} from "@/components/ootd/RecommendationResult";
import Footer from "@/components/landing/Footer";
import UploadFormDialog, {
  type UploadFormData,
} from "@/components/ootd/UploadFormDialog";

import type { ClosetItemView } from "@/lib/types/closet-view";

type SelectedItemsByCategory = {
  top: ClosetItemView[];
  bottom: ClosetItemView[];
  dress: ClosetItemView[];
  outer: ClosetItemView[];
};

export default function OotdPage() {
  const [moodText, setMoodText] = useState("");
  const [ootdFile, setOotdFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [selectedClosetItemId, setSelectedClosetItemId] = useState<
    string | null
  >(null);
  const [recommendationResults, setRecommendationResults] = useState<
    RecommendationItem[]
  >([]);
  const [selectedItems, setSelectedItems] =
    useState<SelectedItemsByCategory | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [recommendationId, setRecommendationId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [newlyUploadedId, setNewlyUploadedId] = useState<string | null>(null);
  const [showUploadForm, setShowUploadForm] = useState(false);

  // 옷장 데이터 상태
  const [closetItems, setClosetItems] = useState<ClosetItemView[]>([]);
  const [closetLoading, setClosetLoading] = useState(true);
  const [closetError, setClosetError] = useState<string | null>(null);

  // 날씨 데이터 (추천 요청에 전달)
  const [weather, setWeather] = useState<{
    temperature?: number;
    feelsLike?: number;
    precipitation?: number;
  }>({});

  // 옷장 데이터 로드
  const fetchCloset = useCallback(async () => {
    try {
      setClosetLoading(true);
      setClosetError(null);
      const res = await fetch("/api/closet");
      if (!res.ok) throw new Error("옷장 데이터 로드 실패");
      const data = await res.json();

      // API 응답의 ClosetItem을 ClosetItemView로 변환
      const views: ClosetItemView[] = data.items.map((item: any) => ({
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

      setClosetItems(views);
    } catch (err) {
      console.error("Closet fetch error:", err);
      setClosetError("옷장 데이터를 불러올 수 없습니다.");
    } finally {
      setClosetLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCloset();
  }, [fetchCloset]);

  // 날씨 데이터 로드
  useEffect(() => {
    async function fetchWeather() {
      try {
        const res = await fetch("/api/weather");
        if (res.ok) {
          const data = await res.json();
          setWeather({
            temperature: data.temperature,
            feelsLike: data.feelsLike,
            precipitation: data.precipitation,
          });
        }
      } catch {
        // 날씨 로드 실패는 무시
      }
    }
    fetchWeather();
  }, []);

  function handleUpload(file: File) {
    setOotdFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }

  function handleRemoveImage() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setOotdFile(null);
    setPreviewUrl("");
  }

  function handleReset() {
    handleRemoveImage();
    setMoodText("");
    setSelectedClosetItemId(null);
    setRecommendationResults([]);
    setSelectedItems(null);
    setRecommendationId(null);
    toast.info("초기화되었습니다.");
  }

  async function handleRecommend() {
    if (moodText.length <= 2) {
      toast.warning("mood를 3자 이상 입력해주세요.");
      return;
    }

    setIsLoading(true);
    setRecommendationResults([]);
    setSelectedItems(null);

    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mood: moodText,
          temperature: weather.temperature,
          feelsLike: weather.feelsLike,
          precipitation: weather.precipitation,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || "추천 요청 실패");
      }

      const data = await res.json();

      // API 응답을 RecommendationItem으로 변환
      const results: RecommendationItem[] = (data.recommendations || [])
        .map((rec: any) => {
          if (rec.type === "dress" && rec.dress) {
            return {
              id: rec.id,
              type: "dress" as const,
              dress: toView(rec.dress),
              outer: rec.outer ? toView(rec.outer) : undefined,
              score: rec.score,
              reason: rec.reason,
            };
          }

          if (!rec.top || !rec.bottom) return null;

          return {
            id: rec.id,
            type: "two_piece" as const,
            top: toView(rec.top),
            bottom: toView(rec.bottom),
            outer: rec.outer ? toView(rec.outer) : undefined,
            score: rec.score,
            reason: rec.reason,
          };
        })
        .filter((item: RecommendationItem | null): item is RecommendationItem => item !== null);

      // Parse selected items (Step 1 results)
      if (data.selectedItems) {
        const si = data.selectedItems;
        setSelectedItems({
          top: (si.top || []).map(toView),
          bottom: (si.bottom || []).map(toView),
          dress: (si.dress || []).map(toView),
          outer: (si.outer || []).map(toView),
        });
      }

      setRecommendationId(data.recommendation_id || null);
      setRecommendationResults(results);
      toast.success("코디 추천이 완료되었습니다!", {
        description: `${results.length}개의 코디를 추천했습니다.`,
        duration: 3000,
      });
    } catch (err) {
      console.error("Recommend error:", err);
      toast.error("코디 추천에 실패했습니다.", {
        description: err instanceof Error ? err.message : "다시 시도해주세요.",
      });
    } finally {
      setIsLoading(false);
    }
  }

  async function handleUploadToCloset(formData: UploadFormData) {
    if (!ootdFile || isUploading) return;

    setIsUploading(true);
    try {
      const body = new FormData();
      body.append("image", ootdFile);
      body.append("category", formData.category);
      body.append("color", formData.color);
      if (formData.subType) body.append("sub_type", formData.subType);

      const res = await fetch("/api/closet/upload", {
        method: "POST",
        body,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || "업로드 실패");
      }

      const data = await res.json();

      toast.success("옷장에 아이템이 추가되었습니다!", {
        description: data.item?.name
          ? `"${data.item.name}" 등록 완료`
          : undefined,
      });

      // 새로 업로드된 아이템 하이라이트
      if (data.item?.id) {
        setNewlyUploadedId(data.item.id);
        setTimeout(() => setNewlyUploadedId(null), 3000);
      }

      setShowUploadForm(false);
      handleRemoveImage();
      fetchCloset(); // 옷장 새로고침
    } catch (err) {
      console.error("Upload error:", err);
      toast.error("업로드에 실패했습니다.", {
        description: err instanceof Error ? err.message : "다시 시도해주세요.",
      });
    } finally {
      setIsUploading(false);
    }
  }

  async function handleFeedback(_resultId: string, isPositive: boolean) {
    if (!recommendationId) {
      toast.warning("추천 ID가 없어 피드백을 저장할 수 없습니다.");
      return;
    }
    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recommendation_id: recommendationId,
          liked: isPositive,
        }),
      });
      if (!res.ok) throw new Error("피드백 저장 실패");
      toast.success(
        isPositive ? "피드백 감사합니다!" : "피드백이 반영되었습니다.",
        { description: "다음 추천에 반영하겠습니다.", duration: 2000 }
      );
    } catch {
      toast.error("피드백 저장에 실패했습니다.");
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 relative overflow-hidden">
      {/* Background Decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-200/20 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-purple-200/20 rounded-full blur-3xl animate-float delay-1000" />
      </div>

      <OotdHeader />

      <div className="pt-16 min-h-screen">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
          {/* Page Title */}
          <div className="mb-8 lg:mb-12 text-center lg:text-left">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-100/50 border border-blue-200/50 text-blue-700 text-sm font-medium mb-4">
              <Wand2 className="size-4" />
              AI 기반 스마트 추천
            </div>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-4">
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                AI 코디 추천
              </span>
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto lg:mx-0">
              날씨와 상황에 맞는 완벽한 코디를 추천받아보세요
            </p>
          </div>

          {/* 추천 입력 영역 */}
          <Card className="overflow-hidden border-border/40 notion-shadow-xl backdrop-blur-sm bg-white/90 relative z-10 mb-8">
            <CardContent className="p-0">
              <div className="grid grid-cols-1 lg:grid-cols-[420px_1fr]">
                {/* ===================== LEFT COLUMN ===================== */}
                <div className="space-y-6 border-b border-border/40 bg-gradient-to-br from-slate-50/90 to-blue-50/50 p-8 lg:border-r lg:border-b-0">
                  {/* [A] Date & Weather */}
                  <DateWeatherHeader />

                  {/* [B] OOTD Upload Card */}
                  <OotdUploadCard
                    previewUrl={previewUrl}
                    onUpload={handleUpload}
                    onRemove={handleRemoveImage}
                  />

                  {/* Upload to closet button */}
                  {ootdFile && (
                    <Button
                      variant="default"
                      size="sm"
                      className="w-full rounded-xl gap-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-md"
                      onClick={() => setShowUploadForm(true)}
                    >
                      <Plus className="size-4" />
                      옷장에 추가하기
                    </Button>
                  )}

                </div>

                {/* ===================== RIGHT COLUMN ===================== */}
                <div className="flex flex-col gap-6 p-8 bg-gradient-to-br from-white/80 to-slate-50/50">
                  {/* [D] Mood Input */}
                  <MoodInput value={moodText} onChange={setMoodText} />

                  {/* [E] Closet Grid */}
                  <div className="flex-1 overflow-y-auto max-h-[600px] pr-2">
                    {closetLoading ? (
                      <div className="flex items-center justify-center py-12">
                        <Loader2 className="size-6 animate-spin text-muted-foreground" />
                        <span className="ml-2 text-sm text-muted-foreground">
                          옷장 로딩 중...
                        </span>
                      </div>
                    ) : closetError ? (
                      <div className="text-center py-12">
                        <p className="text-sm text-red-500">{closetError}</p>
                        <Button
                          variant="outline"
                          size="sm"
                          className="mt-2"
                          onClick={fetchCloset}
                        >
                          다시 시도
                        </Button>
                      </div>
                    ) : (
                      <ClosetGrid
                        items={closetItems}
                        selectedId={selectedClosetItemId}
                        onSelect={setSelectedClosetItemId}
                        highlightId={newlyUploadedId}
                      />
                    )}
                  </div>

                  {/* [F] Action Buttons */}
                  <div className="flex items-center gap-3 pt-4 border-t border-border/40">
                    <Button
                      onClick={handleRecommend}
                      size="lg"
                      disabled={isLoading}
                      className="flex-1 gap-2 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground disabled:opacity-50"
                    >
                      <Wand2 className="size-4" />
                      {isLoading ? "추천 중..." : "추천받기"}
                    </Button>
                    <Button
                      variant="outline"
                      size="lg"
                      onClick={handleReset}
                      disabled={isLoading}
                      className="gap-2 rounded-xl"
                    >
                      <RotateCcw className="size-4" />
                      초기화
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* AI 선정 아이템 영역 (Step 1) */}
          {selectedItems && (
            <Card className="overflow-hidden border-border/40 notion-shadow-xl backdrop-blur-sm bg-white/90 relative z-10 mb-8">
              <CardContent className="p-8">
                <SelectedItemsGrid items={selectedItems} />
              </CardContent>
            </Card>
          )}

          {/* 추천 결과 영역 (Step 4 최종 코디) */}
          {(isLoading || recommendationResults.length > 0) && (
            <Card className="overflow-hidden border-border/40 notion-shadow-xl backdrop-blur-sm bg-white/90 relative z-10">
              <CardContent className="p-8">
                <RecommendationResult
                  results={recommendationResults}
                  loading={isLoading}
                  onFeedback={handleFeedback}
                />
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      <UploadFormDialog
        open={showUploadForm}
        onOpenChange={setShowUploadForm}
        previewUrl={previewUrl}
        loading={isUploading}
        onSubmit={handleUploadToCloset}
      />

      <Footer />
    </div>
  );
}

/**
 * API 응답 아이템을 ClosetItemView로 변환 (추천 결과용)
 */
function toView(item: any): ClosetItemView {
  if (!item) {
    return {
      id: "unknown",
      name: "Unknown",
      imageUrl: "https://picsum.photos/400/500",
      category: "top",
      createdAt: new Date().toISOString().split("T")[0],
    };
  }
  return {
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
    createdAt: item.createdAt || new Date().toISOString().split("T")[0],
  };
}
