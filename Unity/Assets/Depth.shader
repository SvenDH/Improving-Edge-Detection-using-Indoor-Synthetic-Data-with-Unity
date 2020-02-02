Shader "Depth"
{
	SubShader
	{
		Tags{ "RenderType" = "Opaque" }
		Pass
	{
	CGPROGRAM

	#pragma vertex vert
	#pragma fragment frag
	#include "UnityCG.cginc"

	struct v2f {
		float4 pos : SV_POSITION;
		float4 worldPos : NORMAL;
	};

	float3 projectOnVector(float3 B, float3 A) {
		return dot(A, B) / length(A);
	}

	v2f vert(float4 vertex : POSITION, float3 normal : NORMAL) {
		v2f o;
		o.pos = UnityObjectToClipPos(vertex);
		o.worldPos = mul(unity_ObjectToWorld, vertex);
		return o;
	}

	float frag(v2f i) : SV_Target{
		float camFarPlane = _ProjectionParams.z * 1.75;

		float3 viewDir = UNITY_MATRIX_V[2].xyz;
		viewDir = normalize(viewDir);

		float dist = length(projectOnVector(i.worldPos - _WorldSpaceCameraPos, viewDir));
		//return EncodeFloatRGBA(dist / camFarPlane).xyzw;
		//float f = clamp(dist / camFarPlane, 0, 1);
		//float r = clamp(int(f), 0, 255);
		//float g = clamp(int(2 ^ 8 * f), 0, 255);
		//float b = clamp(int(2 ^ 16 * f), 0, 255);
		//return vec4(r / 255, g / 255, b / 255, 0);
		return dist / camFarPlane;
	}

	ENDCG
}
	}
}
