"use client";

import { Cloud, Droplets, MapPin, Loader2 } from "lucide-react";
import { useState, useEffect } from "react";

const DAY_NAMES = ["일", "월", "화", "수", "목", "금", "토"];

type WeatherData = {
  location: string;
  temperature: number;
  feelsLike: number;
  weather: string;
  precipitation: number;
};

// 날씨 타입에 따른 아이콘 매핑
const getWeatherIcon = (weather: string) => {
  const lowerWeather = weather.toLowerCase();
  if (lowerWeather.includes("mist") || lowerWeather.includes("fog")) {
    return <Cloud className="size-5 text-slate-400" />;
  }
  if (lowerWeather.includes("rain") || lowerWeather.includes("drizzle")) {
    return <Droplets className="size-5 text-blue-500" />;
  }
  return <Cloud className="size-5 text-slate-500" />;
};

export default function DateWeatherHeader() {
  const [location] = useState("Chongdong"); // 추후 사용자 선택 기능 추가 가능
  const [weatherData, setWeatherData] = useState<WeatherData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const now = new Date();
  const month = now.getMonth() + 1;
  const date = now.getDate();
  const day = DAY_NAMES[now.getDay()];

  useEffect(() => {
    async function fetchWeatherData() {
      try {
        setIsLoading(true);
        setError(null);

        // API 엔드포인트 - 환경 변수에서 가져오거나 기본값 사용
        const apiUrl = process.env.NEXT_PUBLIC_WEATHER_API_URL || "/api/weather";
        
        const response = await fetch(`${apiUrl}?location=${encodeURIComponent(location)}`);
        
        if (!response.ok) {
          throw new Error("날씨 데이터를 가져오는데 실패했습니다.");
        }

        const data = await response.json();
        
        // 에러 응답 처리
        if (data.error) {
          throw new Error(data.details || data.error);
        }
        
        // API 응답 형식에 맞게 데이터 변환 (실시간 데이터)
        setWeatherData({
          location: data.location || location,
          temperature: data.temperature || 0,
          feelsLike: data.feelsLike || 0,
          weather: data.weather || "Clear",
          precipitation: data.precipitation || 0.0,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "날씨 정보를 불러올 수 없습니다.");
        console.error("Weather API error:", err);
      } finally {
        setIsLoading(false);
      }
    }

    fetchWeatherData();
  }, [location]);

  return (
    <div className="space-y-3">
      {/* 날짜 */}
      <div className="flex items-center justify-between rounded-xl bg-gradient-to-r from-blue-50/50 to-purple-50/30 p-4 border border-border/40">
        <span className="text-xl font-semibold text-foreground tracking-tight">
          {month}/{date} ({day})
        </span>
      </div>

      {/* 날씨 정보 */}
      <div className="rounded-xl bg-gradient-to-br from-slate-50/80 to-blue-50/40 p-4 border border-border/40 space-y-3">
        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="size-5 animate-spin text-blue-600" />
            <span className="ml-2 text-sm text-muted-foreground">날씨 정보 로딩 중...</span>
          </div>
        ) : error ? (
          <div className="text-center py-4">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        ) : weatherData ? (
          <>
            {/* 위치 */}
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <MapPin className="size-4 text-blue-600" />
              <span className="font-medium text-foreground/80">현재 위치: {weatherData.location}</span>
            </div>

            {/* 기온 정보 */}
            <div className="grid grid-cols-2 gap-3">
              <div className="flex items-center gap-2">
                <div className="flex-1">
                  <div className="text-xs text-muted-foreground mb-0.5">현재 기온</div>
                  <div className="text-lg font-bold text-foreground">
                    {weatherData.temperature.toFixed(1)}°C
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex-1">
                  <div className="text-xs text-muted-foreground mb-0.5">체감</div>
                  <div className="text-lg font-bold text-foreground">
                    {weatherData.feelsLike.toFixed(1)}°C
                  </div>
                </div>
              </div>
            </div>

            {/* 날씨 및 강수 */}
            <div className="flex items-center justify-between pt-2 border-t border-border/30">
              <div className="flex items-center gap-2">
                {getWeatherIcon(weatherData.weather)}
                <div>
                  <div className="text-xs text-muted-foreground">날씨</div>
                  <div className="text-sm font-semibold text-foreground">
                    {weatherData.weather}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-muted-foreground">강수</div>
                <div className="text-sm font-semibold text-foreground">
                  {weatherData.precipitation.toFixed(1)}mm
                </div>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
