"use client";

import {
  Wand2,
  Cloud,
  Sparkles,
  Shirt,
  TrendingUp,
  Zap,
} from "lucide-react";

const features = [
  {
    icon: Wand2,
    title: "AI 코디 추천",
    description:
      "날씨, 상황, 기분에 맞춘 개인화된 코디를 AI가 실시간으로 추천합니다.",
    color: "from-blue-500 to-cyan-500",
  },
  {
    icon: Cloud,
    title: "날씨 기반 추천",
    description:
      "실시간 날씨 정보를 반영하여 계절과 온도에 맞는 옷차림을 제안합니다.",
    color: "from-purple-500 to-pink-500",
  },
  {
    icon: Shirt,
    title: "디지털 옷장",
    description:
      "내 옷장을 디지털로 관리하고, 보유한 아이템 기반으로 코디를 구성합니다.",
    color: "from-orange-500 to-red-500",
  },
  {
    icon: Sparkles,
    title: "스타일 분석",
    description:
      "당신의 스타일 선호도를 학습하여 더 정확한 추천을 제공합니다.",
    color: "from-green-500 to-emerald-500",
  },
  {
    icon: TrendingUp,
    title: "트렌드 반영",
    description:
      "최신 패션 트렌드를 반영한 코디 아이디어를 제공합니다.",
    color: "from-indigo-500 to-blue-500",
  },
  {
    icon: Zap,
    title: "빠른 추천",
    description:
      "몇 초 만에 완벽한 코디를 추천받고 바로 적용해보세요.",
    color: "from-yellow-500 to-orange-500",
  },
];

export default function FeaturesSection() {
  return (
    <section
      id="features"
      className="py-24 bg-gradient-to-b from-white via-blue-50/30 to-white"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">
            강력한 기능들
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            AI 기술과 사용자 경험을 결합한 혁신적인 코디 추천 서비스
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div
                key={index}
                className="group relative rounded-2xl bg-white p-8 border border-border/40 notion-shadow notion-hover overflow-hidden"
              >
                {/* Gradient Background on Hover */}
                <div
                  className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-5 transition-opacity duration-300`}
                />

                <div
                  className={`inline-flex rounded-2xl bg-gradient-to-br ${feature.color} p-3 mb-4`}
                >
                  <Icon className="size-6 text-white" />
                </div>

                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
