# TODO 리스트

## 🔴 긴급 (이번 주)

### 데이터베이스 구축
- [ ] PostgreSQL 데이터베이스 생성 (Supabase 추천)
- [ ] `database/schema.sql` 마이그레이션 실행
- [ ] PostgreSQL 연결 테스트
- [ ] `src/lib/db/postgres-repository.ts` 구현
- [ ] Vector DB 선택 및 계정 생성 (Pinecone 추천)
- [ ] Vector DB 인덱스 생성
- [ ] `src/lib/vector-db/client.ts` 실제 구현

### 팀원 협의
- [ ] 모델 서버 API 스펙 문서 작성
- [ ] 팀원과 API 스펙 협의
- [ ] 벡터 차원 수 확인
- [ ] 모델 서버 URL 확인

---

## 🟡 중요 (다음 주)

### 모델 서버 통합
- [ ] 이미지 분석 모델 서버 연동
- [ ] CLIP Image Encoder 연동
- [ ] CLIP Text Encoder 연동
- [ ] 추천 모델 서버 연동
- [ ] 에러 처리 및 재시도 로직

### 이미지 스토리지
- [ ] Cloudinary 계정 생성 (또는 S3/Vercel Blob)
- [ ] 이미지 업로드 API 구현
- [ ] 환경 변수 설정

---

## 🟢 일반 (이후)

### 옷장 관리 완성
- [ ] 프론트엔드 업로드 UI 연동
- [ ] 옷장 조회 API 프론트엔드 연동
- [ ] 아이템 수정/삭제 기능

### 추천 기능 완성
- [ ] Vector DB 검색 구현
- [ ] 프론트엔드 추천 API 호출
- [ ] 추천 결과 표시

### 추가 기능
- [ ] 피드백 API
- [ ] 추천 히스토리
- [ ] 사용자 인증 (선택)

---

## 📝 체크리스트

### 환경 설정
- [ ] `.env.local` 파일 생성
- [ ] 모든 환경 변수 설정
- [ ] 로컬 개발 환경 테스트

### 테스트
- [ ] 옷장 업로드 테스트
- [ ] Vector DB 저장/검색 테스트
- [ ] 추천 API 테스트
- [ ] 전체 플로우 통합 테스트

### 배포
- [ ] Vercel 배포 설정
- [ ] 환경 변수 설정 (Vercel)
- [ ] 데이터베이스 연결 확인
- [ ] Vector DB 연결 확인

---

## 🐛 알려진 이슈

- [ ] `InMemoryClosetRepository`는 임시 구현, 실제 DB로 교체 필요
- [ ] `dummy.json` 로딩은 개발용, 실제로는 Vector DB 사용
- [ ] 모델 서버 URL은 환경 변수로 관리 필요

---

## 💡 향후 개선 사항

- [ ] 캐싱 전략 (Redis 등)
- [ ] 로깅 및 모니터링
- [ ] 성능 최적화
- [ ] 에러 핸들링 개선
