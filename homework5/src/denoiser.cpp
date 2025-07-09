#include "denoiser.h"

Denoiser::Denoiser() : m_useTemportal(false) {}

void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
    Matrix4x4 preWorldToCamera =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Reproject
            m_valid(x, y) = false;
            m_misc(x, y) = Float3(0.f);

            int id = frameInfo.m_id(x, y);
            if (id == -1) {
                continue;
            }
            Matrix4x4 worldToLocal = Inverse(frameInfo.m_matrix[id]);
            Matrix4x4 preLocalToWorld = m_preFrameInfo.m_matrix[id];
            Float3 world_position = frameInfo.m_position(x, y);
            Float3 local_position = worldToLocal(world_position, Float3::EType::Point);
            Float3 pre_world_position = preLocalToWorld(local_position, Float3::EType::Point);
            Float3 pre_screen_position = preWorldToScreen(pre_world_position, Float3::EType::Point);

            if (pre_screen_position.x < 0 || pre_screen_position.x >= width ||
                pre_screen_position.y < 0 || pre_screen_position.y >= height) {
                continue;   // Previous frame position is out of screen bounds
            } else {
                int pre_id = m_preFrameInfo.m_id(pre_screen_position.x, pre_screen_position.y);
                // Check previous frame ?= current frame
                if (pre_id == id) {
                    m_valid(x, y) = true;
                    m_misc(x, y) = m_accColor(pre_screen_position.x, pre_screen_position.y);
                }
            }
        }
    }
    std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 3;
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Temporal clamp
            Float3 color = m_accColor(x, y);
            //Set Alpha to 1 when no legal corresponding point was found in the previous frame
            float alpha = 1.0f;

            if (m_valid(x, y)) {
                alpha = m_alpha;

                int x_start = std::max(0, x - kernelRadius);
                int x_end = std::min(width - 1, x + kernelRadius);
                int y_start = std::max(0, y - kernelRadius);
                int y_end = std::min(height - 1, y + kernelRadius);

                Float3 mu(0.f);
                Float3 sigma(0.f);

                for (int m = x_start; m <= x_end; m++) {
                    for (int n = y_start; n <= y_end; n++) {
                        mu += curFilteredColor(m, n);
                        sigma += Sqr(curFilteredColor(x, y) - curFilteredColor(m, n));
                    }
                }

                int count = kernelRadius * 2 + 1;
                // 7 * 7
                count *= count;

                mu /= float(count);
                sigma = SafeSqrt(sigma / float(count));
                color = Clamp(color, mu - sigma * m_colorBoxK, mu + sigma * m_colorBoxK);
            }

            m_misc(x, y) = Lerp(color, curFilteredColor(x, y), alpha);
        }
    }
    std::swap(m_misc, m_accColor);
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 16;
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Joint bilateral filter
            filteredImage(x, y) = Float3(0.0f);

            int x_start = std::max(0, x - kernelRadius);
            int x_end = std::min(width - 1, x + kernelRadius);
            int y_start = std::max(0, y - kernelRadius);
            int y_end = std::min(height - 1, y + kernelRadius);

            Float3 center_postion = frameInfo.m_position(x, y);
            Float3 center_normal = frameInfo.m_normal(x, y);
            Float3 center_color = frameInfo.m_beauty(x, y);

            Float3 final_color;
            float total_weight = .0f;

            for (int m = x_start; m <= x_end; m++) {
                for (int n = y_start; n <= y_end; n++) {

                    Float3 postion = frameInfo.m_position(m, n);
                    Float3 normal = frameInfo.m_normal(m, n);
                    Float3 color = frameInfo.m_beauty(m, n);

                    //The smaller the distance, the greater the contribution
                    float dist_position = SqrDistance(center_postion, postion) /
                                            (2.0f * m_sigmaCoord * m_sigmaCoord);

                    //The smaller the color difference, the greater the contribution
                    float dist_color = SqrDistance(center_color, color) /
                                        (2.0f * m_sigmaColor * m_sigmaColor);
                    //d_normal : The smaller the angle between normals 1 and 2, the greater the contribution
                    float d_normal = SafeAcos(Dot(center_normal, normal));
                    float dist_normal = d_normal * d_normal / (2.0f * m_sigmaNormal * m_sigmaNormal);

                    //d_plane : The smaller the value of center_normal dot Normalize(postion - center_postion), the greater the contribution
                    float d_plane = Dot(center_normal, Normalize(postion - center_postion));
                    float dist_plane = d_plane * d_plane / (2.0f * m_sigmaPlane * m_sigmaPlane);

                    float weight = std::exp(- dist_position - dist_color - dist_normal - dist_plane);
                    total_weight += weight;
                    final_color += color * weight;
                }
            }
            if (total_weight == 0)
                filteredImage(x, y) = center_color;
            else
                filteredImage(x, y) = final_color / total_weight;
        }
    }
    return filteredImage;
}

void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
    m_accColor.Copy(filteredColor);
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    m_misc = CreateBuffer2D<Float3>(width, height);
    m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }

Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
    filteredColor = Filter(frameInfo);

    // Reproject previous frame color to current
    if (m_useTemportal) {
        Reprojection(frameInfo);
        TemporalAccumulation(filteredColor);
    } else {
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}
