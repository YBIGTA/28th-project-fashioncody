"use client";

import { ArrowRight, Sparkles, Wand2, ImageIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-gradient-to-br from-blue-50 via-white to-purple-50 pt-16">
      {/* Background Decoration */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-200/30 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-purple-200/30 rounded-full blur-3xl animate-float delay-1000" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left: Text Content */}
          <div className="space-y-8 text-center lg:text-left">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-100/50 border border-blue-200/50 text-blue-700 text-sm font-medium">
              <Sparkles className="size-4" />
              AI 기반 스마트 코디 추천
            </div>

            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight">
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                나만의
              </span>
              <br />
              <span className="text-foreground">스타일을 찾아보세요</span>
            </h1>

            <p className="text-xl text-muted-foreground max-w-2xl mx-auto lg:mx-0">
              AI가 당신의 옷장을 분석하고, 날씨와 상황에 맞는 완벽한 코디를
              추천해드립니다. 매일 새로운 스타일을 발견하세요.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Link href="/ootd">
                <Button size="lg" className="rounded-xl gap-2 group">
                  무료로 시작하기
                  <ArrowRight className="size-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
              <Button
                variant="outline"
                size="lg"
                className="rounded-xl gap-2"
              >
                <Wand2 className="size-4" />
                데모 보기
              </Button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-6 pt-8 border-t border-border/40">
              <div>
                <div className="text-3xl font-bold text-foreground">10K+</div>
                <div className="text-sm text-muted-foreground">활성 사용자</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-foreground">50K+</div>
                <div className="text-sm text-muted-foreground">코디 추천</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-foreground">4.9★</div>
                <div className="text-sm text-muted-foreground">평균 평점</div>
              </div>
            </div>
          </div>

          {/* Right: Visual */}
          <div className="relative">
            <div className="relative rounded-3xl overflow-hidden notion-shadow-xl bg-white p-8 border border-border/40">
              <div className="aspect-[3/4] rounded-2xl bg-gradient-to-br from-blue-100 to-purple-100 flex items-center justify-center">
                <ImageIcon className="size-24 text-blue-300" />
              </div>
              {/* Floating Cards */}
              <div className="absolute -top-4 -right-4 rounded-2xl bg-white p-4 notion-shadow-lg border border-border/40 animate-bounce">
                <Wand2 className="size-6 text-blue-600" />
              </div>
              <div className="absolute -bottom-4 -left-4 rounded-2xl bg-white p-4 notion-shadow-lg border border-border/40 animate-bounce delay-500">
                <Sparkles className="size-6 text-purple-600" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
