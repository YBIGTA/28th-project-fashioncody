"use client";

import { ArrowRight, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function CTASection() {
  return (
    <section className="py-24 bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-0 left-0 w-full h-full bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4xIj48Y2lyY2xlIGN4PSIzMCIgY3k9IjMwIiByPSIyIi8+PC9nPjwvZz48L3N2Zz4=')]" />
      </div>

      <div className="relative z-10 mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/20 backdrop-blur-sm border border-white/30 text-white text-sm font-medium mb-6">
          <Sparkles className="size-4" />
          지금 바로 시작하세요
        </div>

        <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-6">
          당신의 스타일 여정을
          <br />
          시작해보세요
        </h2>

        <p className="text-xl text-white/90 mb-8 max-w-2xl mx-auto">
          AI 기반 코디 추천으로 매일 새로운 스타일을 발견하고, 자신만의
          패션을 완성해보세요.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link href="/ootd">
            <Button
              size="lg"
              className="rounded-xl gap-2 bg-white text-blue-600 hover:bg-white/90 group"
            >
              무료로 시작하기
              <ArrowRight className="size-4 group-hover:translate-x-1 transition-transform" />
            </Button>
          </Link>
          <Button
            variant="outline"
            size="lg"
            className="rounded-xl gap-2 border-white/30 text-white hover:bg-white/10"
          >
            더 알아보기
          </Button>
        </div>
      </div>
    </section>
  );
}
