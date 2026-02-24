import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const location = searchParams.get("location") || "Chongdong";

    // 환경 변수에서 API 키 가져오기 (WEATHERAPI_KEY 또는 WEATHER_API_KEY 지원)
    const apiKey = process.env.WEATHERAPI_KEY || process.env.WEATHER_API_KEY;
    
    // API 키가 없으면 에러 반환
    if (!apiKey) {
      return NextResponse.json(
        { 
          error: "날씨 API 키가 설정되지 않았습니다.",
          details: "WEATHERAPI_KEY 환경 변수를 설정해주세요."
        },
        { status: 500 }
      );
    }

    // WeatherAPI.com API 호출
    const apiUrl = `http://api.weatherapi.com/v1/current.json`;
    
    const response = await fetch(
      `${apiUrl}?key=${apiKey}&q=${encodeURIComponent(location)}&lang=ko`
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("WeatherAPI.com error:", response.status, errorText);
      return NextResponse.json(
        { 
          error: "날씨 정보를 가져오는데 실패했습니다.",
          details: `API Error: ${response.status} ${response.statusText}`
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    
    // API 응답 로깅 (개발용)
    if (process.env.NODE_ENV === "development") {
      console.log("WeatherAPI.com Response:", JSON.stringify(data, null, 2));
    }

    // WeatherAPI.com 응답 형식에 맞게 변환하여 반환
    return NextResponse.json({
      location: data.location?.name || location,
      temperature: data.current?.temp_c || 0,
      feelsLike: data.current?.feelslike_c || 0,
      weather: data.current?.condition?.text || "Clear",
      precipitation: data.current?.precip_mm || 0.0,
    });
  } catch (error) {
    console.error("Weather API error:", error);
    return NextResponse.json(
      { 
        error: "날씨 정보를 가져오는데 실패했습니다.",
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}
